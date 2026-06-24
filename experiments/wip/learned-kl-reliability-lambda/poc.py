"""
PoC #1 (die point) for `learned-reliability-weighted-λ-return`.

Question (advisor-hardened):
  Does a per-step reliability signal predict *decision-relevant* imagination drift
  in a DreamerV3-style RSSM, BEYOND the trivial horizon-time trend, well enough
  that reweighting the λ-return target by w=exp(-β·signal) reduces target error?

Design (4 advisor corrections baked in):
  1. t-CONTROLLED correlation: gate on within-step (fixed horizon t) rank correlation
     across trajectories, averaged over t. Pooled corr reported only for contrast.
     (A signal that tracks drift only through t merely re-discovers the discount.)
  2. DECISION-RELEVANT drift: drift_t = |r_imag-r_post| + |v_imag-v_post| through the
     SAME reward/critic heads (the only difference is the latent). Latent-L2 reported
     for the chain but NOT gated (latent-L2 that misses value harm = the BL-11 trap).
  3. SPEARMAN (rank), not Pearson: w=exp(-β·s) needs monotonicity; drift is heavy-tailed.
  4. Correlation is the SCREEN, not GO. Real feasibility signal = does reweighting by w
     actually reduce |λ-target - MC_return_from_real_rewards|?

Setup: GL-free, fully synthetic. A heteroscedastic, partially-observed nonlinear env
gives STATE-DEPENDENT predictability (some regions drift more at the SAME horizon t),
which is exactly what makes the t-controlled test meaningful. Models are a compact but
faithful DreamerV3 RSSM (GRU deter + Gaussian stochastic, prior=img_step / posterior=
obs_step, KL(post||prior)) + reward head + two-hot categorical critic.

This is the validator-stage synthetic gate. The real DMControl experiment is the next stage.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sps

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------
# Synthetic environment: heteroscedastic, partially observed nonlinear oscillator.
#   true state x=(p,v) in R^2. Two regions by |p|: an "easy" low-noise region and a
#   "hard" high-noise / sensitive region -> state-dependent prior-rollout drift.
#   Observation: position only (+ small noise) -> POMDP, RSSM must integrate.
#   Reward: smooth, -p^2 - 0.1 v^2 (so reward/value heads carry signal).
# ----------------------------------------------------------------------------------
class HeteroOsc:
    def __init__(self, dt=0.1):
        self.dt = dt
    def reset(self, n):
        p = np.random.uniform(-1.5, 1.5, size=n)
        v = np.random.uniform(-1.0, 1.0, size=n)
        self.x = np.stack([p, v], 1).astype(np.float32)
        return self._obs()
    def _obs(self):
        # partial: position only, small obs noise
        p = self.x[:, 0]
        return (p + 0.02 * np.random.randn(*p.shape)).astype(np.float32)[:, None]
    def _proc_noise_scale(self, p):
        # HARD (high process noise + chaotic stiffness) when |p|>0.9, EASY otherwise.
        return np.where(np.abs(p) > 0.9, 0.45, 0.03).astype(np.float32)
    def step(self, a):
        a = np.clip(a, -1, 1).reshape(-1)
        p, v = self.x[:, 0], self.x[:, 1]
        # nonlinear restoring force; stiffness rises sharply in the hard region (sensitive)
        stiff = 1.0 + 4.0 * (np.abs(p) > 0.9)
        force = -stiff * np.sin(p) + 0.8 * a
        sig = self._proc_noise_scale(p)
        v2 = v + self.dt * force + sig * np.random.randn(*v.shape).astype(np.float32)
        p2 = p + self.dt * v2
        self.x = np.stack([p2, v2], 1).astype(np.float32)
        r = (-(p2 ** 2) - 0.1 * (v2 ** 2)).astype(np.float32)
        return self._obs(), r[:, None]

def collect(env, n_traj, T):
    """Random-policy rollouts. Returns obs (T+1,N,1), act (T,N,1), rew (T,N,1)."""
    obs0 = env.reset(n_traj)
    obs = [obs0]; acts = []; rews = []
    for t in range(T):
        a = np.random.uniform(-1, 1, size=(n_traj, 1)).astype(np.float32)
        o, r = env.step(a)
        acts.append(a); rews.append(r); obs.append(o)
    obs = np.stack(obs, 0); acts = np.stack(acts, 0); rews = np.stack(rews, 0)
    return (torch.tensor(obs, device=DEV), torch.tensor(acts, device=DEV),
            torch.tensor(rews, device=DEV))

# ----------------------------------------------------------------------------------
# Compact but faithful DreamerV3-style RSSM.
# ----------------------------------------------------------------------------------
DETER = 64; STOCH = 16; HID = 128; OBS = 1; ACT = 1

class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(OBS, HID), nn.ELU(), nn.Linear(HID, HID), nn.ELU())
        self.gru = nn.GRUCell(HID, DETER)
        self.inp = nn.Sequential(nn.Linear(STOCH + ACT, HID), nn.ELU())  # (stoch,act)->gru input
        self.prior_net = nn.Sequential(nn.Linear(DETER, HID), nn.ELU(), nn.Linear(HID, 2 * STOCH))
        self.post_net = nn.Sequential(nn.Linear(DETER + HID, HID), nn.ELU(), nn.Linear(HID, 2 * STOCH))
        self.dec = nn.Sequential(nn.Linear(DETER + STOCH, HID), nn.ELU(), nn.Linear(HID, OBS))

    def initial(self, n):
        return dict(deter=torch.zeros(n, DETER, device=DEV), stoch=torch.zeros(n, STOCH, device=DEV))

    def _stats(self, net, x):
        m, s = torch.chunk(net(x), 2, -1)
        s = F.softplus(s) + 0.1
        return m, s

    def img_step(self, st, a):
        gin = self.inp(torch.cat([st["stoch"], a], -1))
        deter = self.gru(gin, st["deter"])
        m, s = self._stats(self.prior_net, deter)
        dist = torch.distributions.Normal(m, s)
        stoch = dist.rsample()
        return dict(deter=deter, stoch=stoch, mean=m, std=s)

    def obs_step(self, st, a, obs):
        prior = self.img_step(st, a)
        e = self.enc(obs)
        m, s = self._stats(self.post_net, torch.cat([prior["deter"], e], -1))
        dist = torch.distributions.Normal(m, s)
        stoch = dist.rsample()
        post = dict(deter=prior["deter"], stoch=stoch, mean=m, std=s)
        return post, prior

    def feat(self, st):
        return torch.cat([st["deter"], st["stoch"]], -1)

    def observe(self, obs, act):
        """obs (T+1,N,1), act (T,N,1). Returns list of posteriors and priors over t=0..T-1."""
        T = act.shape[0]; n = act.shape[1]
        st = self.initial(n)
        posts, priors = [], []
        for t in range(T):
            post, prior = self.obs_step(st, act[t], obs[t + 1])
            posts.append(post); priors.append(prior)
            st = dict(deter=post["deter"], stoch=post["stoch"])
        return posts, priors

def normal_kl(post, prior):
    p = torch.distributions.Normal(post["mean"], post["std"])
    q = torch.distributions.Normal(prior["mean"], prior["std"])
    return torch.distributions.kl_divergence(p, q).sum(-1)  # (N,)

# ---- two-hot categorical critic (DreamerV3-style) for label-shift-free "spread" ----
VMIN, VMAX, NB = -60.0, 5.0, 51
BINS = torch.linspace(VMIN, VMAX, NB, device=DEV)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DETER + STOCH, HID), nn.ELU(),
                                 nn.Linear(HID, HID), nn.ELU(), nn.Linear(HID, NB))
    def dist(self, feat):
        return torch.softmax(self.net(feat), -1)  # (...,NB)
    def mean(self, feat):
        p = self.dist(feat); return (p * BINS).sum(-1)
    def std(self, feat):
        p = self.dist(feat); m = (p * BINS).sum(-1, keepdim=True)
        var = (p * (BINS - m) ** 2).sum(-1)
        return torch.sqrt(var + 1e-8)

class Reward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DETER + STOCH, HID), nn.ELU(), nn.Linear(HID, 1))
    def forward(self, feat):
        return self.net(feat).squeeze(-1)

def twohot_target(y):
    y = y.clamp(VMIN, VMAX)
    idx = torch.bucketize(y, BINS) - 1
    idx = idx.clamp(0, NB - 2)
    lo = BINS[idx]; hi = BINS[idx + 1]
    w_hi = (y - lo) / (hi - lo + 1e-8); w_lo = 1 - w_hi
    tgt = torch.zeros(*y.shape, NB, device=DEV)
    tgt.scatter_(-1, idx.unsqueeze(-1), w_lo.unsqueeze(-1))
    tgt.scatter_(-1, (idx + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return tgt

GAMMA = 0.99
def mc_returns(rew):
    """Discounted MC return from real rewards (no bootstrap), shape (T,N)."""
    T, n, _ = rew.shape
    r = rew.squeeze(-1)
    G = torch.zeros(T, n, device=DEV); acc = torch.zeros(n, device=DEV)
    for t in reversed(range(T)):
        acc = r[t] + GAMMA * acc
        G[t] = acc
    return G

# ----------------------------------------------------------------------------------
# Correlation helpers (t-controlled is the headline).
# ----------------------------------------------------------------------------------
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return sps.spearmanr(a, b).correlation

def t_controlled_spearman(signal, drift):
    """signal,drift: (H,N). Spearman across trajectories at each fixed horizon h, averaged."""
    H = signal.shape[0]; per_h = []
    for h in range(H):
        per_h.append(spearman(signal[h], drift[h]))
    per_h = np.array(per_h, float)
    return np.nanmean(per_h), per_h

def pooled_spearman(signal, drift):
    return spearman(signal.reshape(-1), drift.reshape(-1))

def partial_spearman_controlling_t(signal, drift):
    """Rank-partial correlation of signal vs drift controlling for horizon index t."""
    H, N = signal.shape
    t_idx = np.repeat(np.arange(H), N).astype(float)
    s = signal.reshape(-1); d = drift.reshape(-1)
    # rank-transform then linearly regress out t, correlate residuals
    def rank(x): return sps.rankdata(x)
    rs, rd, rt = rank(s), rank(d), rank(t_idx)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    es, ed = resid(rs, rt), resid(rd, rt)
    if np.std(es) < 1e-9 or np.std(ed) < 1e-9: return np.nan
    return np.corrcoef(es, ed)[0, 1]

# ----------------------------------------------------------------------------------
# Train world model + reward + critic + KL-predictor.
# ----------------------------------------------------------------------------------
def train(rssm, rew_head, critic, rho_head, obs, act, rew, steps=4000, bs=128):
    params = list(rssm.parameters()) + list(rew_head.parameters())
    opt = torch.optim.Adam(params, 1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), 1e-3)
    opt_r = torch.optim.Adam(rho_head.parameters(), 1e-3)
    Tg, N, _ = act.shape
    Gmc = mc_returns(rew)  # (T,N) real-reward MC return targets for critic
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), device=DEV)
        o = obs[:, idx]; a = act[:, idx]; r = rew[:, idx]
        posts, priors = rssm.observe(o, a)
        feats = torch.stack([rssm.feat(p) for p in posts], 0)   # (T,bs,F)
        # WM losses
        recon = rssm.dec(feats)                                  # predict obs[t+1]
        loss_rec = F.mse_loss(recon, o[1:])
        loss_rew = F.mse_loss(rew_head(feats), r.squeeze(-1))
        kl = torch.stack([normal_kl(posts[t], priors[t]) for t in range(Tg)], 0)
        loss_kl = kl.mean()
        loss_wm = loss_rec + loss_rew + 1.0 * loss_kl
        opt.zero_grad(); loss_wm.backward(); opt.step()
        # critic: two-hot regression to real-reward MC returns on posterior feats
        with torch.no_grad():
            f_det = feats.detach()
            tgt = twohot_target(Gmc[:, idx])
        logp = torch.log(critic.dist(f_det) + 1e-8)
        loss_c = -(tgt * logp).sum(-1).mean()
        opt_c.zero_grad(); loss_c.backward(); opt_c.step()
        # rho head: regress one-step KL(post||prior) from PRIOR feat (detached)
        with torch.no_grad():
            prior_feat = torch.stack([torch.cat([priors[t]["deter"], priors[t]["mean"]], -1)
                                      for t in range(Tg)], 0).detach()
            kl_label = kl.detach()
        rho_pred = rho_head(prior_feat).squeeze(-1)
        loss_rho = F.mse_loss(rho_pred, kl_label)
        opt_r.zero_grad(); loss_rho.backward(); opt_r.step()
        if it % 1000 == 0:
            print(f"[train {it}] rec={loss_rec.item():.3f} rew={loss_rew.item():.3f} "
                  f"kl={loss_kl.item():.3f} critic={loss_c.item():.3f} rho={loss_rho.item():.3f}",
                  flush=True)

# ----------------------------------------------------------------------------------
# PoC #1 analysis: teacher-forced prior-only imagination vs posterior ground truth.
# ----------------------------------------------------------------------------------
def analyze(rssm, rew_head, critic, rho_head, obs, act, rew):
    Tg, N, _ = act.shape
    posts, priors = rssm.observe(obs, act)
    # anchor imagination at post[0], roll forward with REAL actions act[1..T-1]
    H = Tg - 1
    imag = [dict(deter=posts[0]["deter"].detach(), stoch=posts[0]["stoch"].detach())]
    imag_priors = [None]
    st = imag[0]
    for h in range(1, Tg):
        pr = rssm.img_step(st, act[h])
        st = dict(deter=pr["deter"], stoch=pr["stoch"])
        imag.append(st); imag_priors.append(pr)
    # per (h,traj) quantities for h=1..H
    drift = np.zeros((H, N)); latent_drift = np.zeros((H, N))
    sig_spread = np.zeros((H, N)); sig_prior_ent = np.zeros((H, N)); sig_rho = np.zeros((H, N))
    r_imag_arr = np.zeros((H, N)); v_imag_arr = np.zeros((H, N)); v_post_arr = np.zeros((H, N))
    with torch.no_grad():
        for h in range(1, Tg):
            fp = rssm.feat(posts[h]); fi = rssm.feat(imag[h])
            r_post = rew_head(fp); r_imag = rew_head(fi)
            v_post = critic.mean(fp); v_imag = critic.mean(fi)
            drift[h - 1] = (r_imag - r_post).abs().cpu().numpy() + (v_imag - v_post).abs().cpu().numpy()
            latent_drift[h - 1] = (fi - fp).norm(dim=-1).cpu().numpy()
            sig_spread[h - 1] = critic.std(fi).cpu().numpy()
            pr = imag_priors[h]
            ent = 0.5 * torch.log(2 * math.pi * math.e * pr["std"] ** 2).sum(-1)
            sig_prior_ent[h - 1] = ent.cpu().numpy()
            prior_feat = torch.cat([pr["deter"], pr["mean"]], -1)
            sig_rho[h - 1] = rho_head(prior_feat).squeeze(-1).cpu().numpy()
            r_imag_arr[h - 1] = r_imag.cpu().numpy(); v_imag_arr[h - 1] = v_imag.cpu().numpy()
            v_post_arr[h - 1] = v_post.cpu().numpy()

    # ---- sanity: drift variance at fixed t (must be non-degenerate) ----
    drift_cv_per_h = (drift.std(1) / (np.abs(drift.mean(1)) + 1e-8)).tolist()

    res = {}
    for name, sig in [("critic_spread", sig_spread), ("learned_rho", sig_rho),
                      ("prior_entropy_BL08", sig_prior_ent)]:
        tc_mean, per_h = t_controlled_spearman(sig, drift)
        res[name] = dict(
            t_controlled_spearman=float(tc_mean),
            pooled_spearman=float(pooled_spearman(sig, drift)),
            partial_spearman_ctrl_t=float(partial_spearman_controlling_t(sig, drift)),
            per_horizon=[None if np.isnan(x) else float(x) for x in per_h],
        )
    # also: how each signal tracks LATENT drift (chain, not gated)
    res["_latent_drift_check"] = {
        name: float(t_controlled_spearman(sig, latent_drift)[0])
        for name, sig in [("critic_spread", sig_spread), ("learned_rho", sig_rho)]
    }

    # ---- feasibility test: SIGN-UNBIASED selective bootstrap (advisor Run-2 design) ----
    # truth G = sum_k g^(k-1) r_real[k] + g^H v_post[H]  (real reward + ground-truth-state bootstrap)
    # T_imag(m) = sum_{k<=m} g^(k-1) r_imag[k] + g^m v_imag[m]  (bootstrap critic value at flagged step m)
    # Controls: (1) oracle = true drift as signal, (2) state-agnostic uniform-m baseline,
    #           (3) tune/test split for hyperparam (tau / m) selection -> no cherry-pick.
    r_real_arr = rew.squeeze(-1).cpu().numpy()[1:Tg]     # (H,N): reward reaching post[1..H]
    gpow = GAMMA ** np.arange(H)                         # (H,)
    G_truth = (gpow[:, None] * r_real_arr).sum(0) + GAMMA ** H * v_post_arr[H - 1]   # (N,)

    def T_of_m(m_arr, r_im=r_imag_arr, v_im=v_imag_arr):
        out = np.zeros(N)
        for n in range(N):
            m = int(m_arr[n])
            out[n] = (gpow[:m] * r_im[:m, n]).sum() + GAMMA ** m * v_im[m - 1, n]
        return out

    def first_flag_m(sig, tau):
        m = np.full(N, H, dtype=int)
        for n in range(N):
            idx = np.where(sig[:, n] > tau)[0]
            if len(idx) > 0:
                m[n] = idx[0] + 1
        return m

    tune = np.arange(0, N // 2); test = np.arange(N // 2, N)
    def err_on(idx, Tvals): return float(np.mean(np.abs(Tvals[idx] - G_truth[idx])))

    # full-rollout baseline (m=H everywhere). sanity: must equal T_imag at m=H.
    T_full = T_of_m(np.full(N, H, int))
    full_test = err_on(test, T_full); full_tune = err_on(tune, T_full)

    # state-agnostic: best uniform m on tune, eval on test
    ag_errs = {m: err_on(tune, T_of_m(np.full(N, m, int))) for m in range(1, H + 1)}
    ag_best_m = min(ag_errs, key=ag_errs.get)
    agnostic_test = err_on(test, T_of_m(np.full(N, ag_best_m, int)))

    def signal_method(sig):
        qs = np.quantile(sig, np.linspace(0.05, 0.95, 19))
        taus = sorted(set(list(qs) + [np.inf]))
        errs = {float(t): err_on(tune, T_of_m(first_flag_m(sig, t))) for t in taus}
        best_t = min(errs, key=errs.get)
        return dict(best_tau=float(best_t),
                    test_err=err_on(test, T_of_m(first_flag_m(sig, best_t))),
                    tune_err=float(errs[best_t]))

    feas = dict(full_test=full_test, full_tune=full_tune,
                agnostic_best_m=int(ag_best_m), agnostic_test=agnostic_test,
                oracle=signal_method(drift),
                critic_spread=signal_method(sig_spread),
                learned_rho=signal_method(sig_rho),
                sanity_full_recovered=bool(abs(full_tune - err_on(tune, T_of_m(first_flag_m(sig_spread, np.inf)))) < 1e-6))
    res["feasibility_selective_bootstrap"] = feas
    res["_sanity_drift_cv_per_h"] = drift_cv_per_h
    return res

def main():
    t0 = time.time()
    env = HeteroOsc()
    print("collecting train/eval trajectories...", flush=True)
    T = 16
    obs_tr, act_tr, rew_tr = collect(env, 1024, T)
    obs_ev, act_ev, rew_ev = collect(HeteroOsc(), 1024, T)
    rssm = RSSM().to(DEV); rew_head = Reward().to(DEV); critic = Critic().to(DEV)
    rho_head = nn.Sequential(nn.Linear(DETER + STOCH, HID), nn.ELU(), nn.Linear(HID, 1)).to(DEV)
    print("training...", flush=True)
    train(rssm, rew_head, critic, rho_head, obs_tr, act_tr, rew_tr, steps=4000, bs=128)
    print("analyzing...", flush=True)
    res = analyze(rssm, rew_head, critic, rho_head, obs_ev, act_ev, rew_ev)

    # ---- verdict logic (PRE-REGISTERED in result_001.md, 2026-06-18 KST) ----
    GATE_CORR = 0.30
    MARGIN = 0.95   # treatment must beat min(full, agnostic) by >=5%
    principled = {k: res[k]["t_controlled_spearman"] for k in ["critic_spread", "learned_rho"]}
    best_sig = max(principled, key=lambda k: (principled[k] if not math.isnan(principled[k]) else -9))
    corr_pass = (not math.isnan(principled[best_sig])) and principled[best_sig] > GATE_CORR

    f = res["feasibility_selective_bootstrap"]
    bar = MARGIN * min(f["full_test"], f["agnostic_test"])
    oracle_pass = f["oracle"]["test_err"] <= bar
    learned_pass = f["critic_spread"]["test_err"] <= bar

    if not f["sanity_full_recovered"]:
        verdict, reason = "INVALID", "sanity: tau=inf did not recover full-rollout baseline (formulation bug)"
    elif corr_pass and oracle_pass and learned_pass:
        verdict, reason = "CONDITIONAL-GO", "signal valid + selective-bootstrap (oracle AND learned) beats full & agnostic by >=5%"
    elif corr_pass and oracle_pass and not learned_pass:
        verdict, reason = "FAIL", "signal-quality: oracle helps but learned signal does not beat baselines (mechanism ok, signal insufficient)"
    elif corr_pass and not oracle_pass:
        verdict, reason = "FAIL", "robust: even ORACLE selective-bootstrap does not beat full/agnostic -> reliability-aware targeting itself does not help here"
    else:
        verdict, reason = "FAIL", "signal validity gate failed"

    res["_verdict"] = dict(
        verdict=verdict, reason=reason, best_signal=best_sig,
        best_t_controlled_spearman=float(principled[best_sig]), gate_corr=GATE_CORR,
        corr_pass=bool(corr_pass), oracle_pass=bool(oracle_pass), learned_pass=bool(learned_pass),
        bar_err=float(bar), full_test=float(f["full_test"]), agnostic_test=float(f["agnostic_test"]),
        oracle_test=float(f["oracle"]["test_err"]), critic_spread_test=float(f["critic_spread"]["test_err"]),
        bl11_precedent=0.085, runtime_s=round(time.time() - t0, 1), device=DEV, seed=SEED,
        note="Pre-registered (result_001.md). Selective bootstrap, sign-unbiased. Treatment must beat full AND agnostic by >=5% on held-out test.")
    with open(os.path.join(OUT, "poc_results.json"), "w") as fo:
        json.dump(res, fo, indent=2)
    print(json.dumps(res["_verdict"], indent=2), flush=True)
    print("\nt-controlled Spearman (headline):", {k: round(principled[k],3) for k in principled}, flush=True)
    print("prior_entropy(BL08) t-ctrl:", round(res["prior_entropy_BL08"]["t_controlled_spearman"],3), flush=True)
    print("feasibility:", json.dumps(f, indent=2), flush=True)

if __name__ == "__main__":
    main()
