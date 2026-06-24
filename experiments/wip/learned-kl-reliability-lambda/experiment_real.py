"""
REAL-DreamerV3 replication of the learned-kl-reliability-lambda PoC gate.

Directly tests the ONE caveat of the synthetic FAIL: "synthetic critic too good ->
bootstrap dominates -> reliability-reweighting pointless; real DreamerV3 may differ."

Loads a vanilla DreamerV3 checkpoint trained on DMControl, then runs the IDENTICAL
pre-registered analysis (result_001.md):
  - signal validity: t-controlled Spearman(signal, decision-relevant drift)
  - feasibility: sign-unbiased selective bootstrap with ORACLE + state-agnostic controls,
    tune/test split, gate = treatment beats full AND agnostic by >=5% on held-out test.

Same verdict logic as poc.py. Run with LD_LIBRARY_PATH=/home/jovyan/egl_libs (EGL).
Usage: python experiment_real.py --logdir <dreamer_logdir>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, json, math, argparse, pathlib
DT = "/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/dreamerv3-torch"
sys.path.insert(0, DT)
import numpy as np
import torch
import torch.distributions as torchd
import yaml
from scipy import stats as sps
import tools, models, networks
from dreamer import make_env, make_dataset, Dreamer
from parallel import Damy

GAMMA = 0.99; OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- numpy helpers (same as poc.py) ----------------
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return np.nan
    return sps.spearmanr(a, b).correlation

def t_controlled_spearman(sig, drift):
    H = sig.shape[0]; per = [spearman(sig[h], drift[h]) for h in range(H)]
    return np.nanmean(per), np.array(per, float)

def pooled_spearman(sig, drift): return spearman(sig.reshape(-1), drift.reshape(-1))

def partial_spearman_controlling_t(sig, drift):
    H, N = sig.shape
    t_idx = np.repeat(np.arange(H), N).astype(float)
    rank = sps.rankdata
    rs, rd, rt = rank(sig.reshape(-1)), rank(drift.reshape(-1)), rank(t_idx)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; b = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ b
    es, ed = resid(rs, rt), resid(rd, rt)
    if np.std(es) < 1e-9 or np.std(ed) < 1e-9: return np.nan
    return float(np.corrcoef(es, ed)[0, 1])

# ---------------- build config exactly like dreamer.py ----------------
def build_config(logdir, device):
    configs = yaml.safe_load((pathlib.Path(DT) / "configs.yaml").read_text())
    def ru(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and k in base: ru(base[k], v)
            else: base[k] = v
    defaults = {}
    for name in ["defaults", "dmc_proprio"]: ru(defaults, configs[name])
    p = argparse.ArgumentParser()
    for k, v in sorted(defaults.items()): p.add_argument(f"--{k}", type=tools.args_type(v), default=tools.args_type(v)(v))
    cfg = p.parse_args([])
    # yaml parses '3e-5' (no dot) as str; args_type only coerces top-level scalars, not nested
    # dict values (actor/critic lr/eps/entropy). Coerce numeric strings -> float so optimizer
    # construction succeeds (we only do inference; lr/eps are irrelevant, arch params unaffected).
    def coerce(d):
        for k, v in list(d.items()):
            if isinstance(v, dict): coerce(v)
            elif isinstance(v, str):
                try: d[k] = float(v)
                except ValueError: pass
    for _k, _v in vars(cfg).items():
        if isinstance(_v, dict): coerce(_v)
    cfg.task = "dmc_walker_walk"; cfg.device = device; cfg.logdir = logdir
    cfg.traindir = pathlib.Path(logdir) / "train_eps"
    cfg.evaldir = pathlib.Path(logdir) / "eval_eps"
    cfg.steps = int(cfg.steps); cfg.act_space = None
    return cfg

def load_agent(cfg):
    train_eps = tools.load_episodes(cfg.traindir, limit=cfg.dataset_size)
    env = Damy(make_env(cfg, "train", 0))
    acts = env.action_space
    cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    ds = make_dataset(train_eps, cfg)
    agent = Dreamer(env.observation_space, env.action_space, cfg, tools.Logger(pathlib.Path(cfg.logdir), 0), ds).to(cfg.device)
    agent.requires_grad_(False)
    ck = torch.load(pathlib.Path(cfg.logdir) / "latest.pt", map_location=cfg.device, weights_only=False)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()
    return agent, ds, env

# ---------------- value-distribution spread (symlog-space std) ----------------
def disc_spread(dist):
    p = dist.probs                      # (...,255)
    b = dist.buckets                    # (255,) symlog space
    m = (p * b).sum(-1, keepdim=True)
    var = (p * (b - m) ** 2).sum(-1)
    return torch.sqrt(var + 1e-8)       # (...,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--nbatches", type=int, default=20)
    a = ap.parse_args()
    DEV = a.device
    cfg = build_config(a.logdir, DEV)
    agent, ds, env = load_agent(cfg)
    wm = agent._wm; dyn = wm.dynamics; critic = agent._task_behavior.value
    T = a.T

    # ---- gather batches, run observe + teacher-forced prior-only imagination ----
    drift=[]; latent=[]; spread=[]; pent=[]; rho_feat=[]; kl_lab=[]
    r_imag=[]; v_imag=[]; v_post=[]; r_real=[]
    with torch.no_grad():
        for _ in range(a.nbatches):
            data = next(ds)
            data = wm.preprocess(data)
            embed = wm.encoder(data)
            post, prior = dyn.observe(embed[:, :T], data["action"][:, :T], data["is_first"][:, :T])
            # feats (B,T,F)
            fp = dyn.get_feat(post)
            # anchor at t=0, imagine forward with REAL actions a[1..T-1]
            anchor = {k: v[:, 0] for k, v in post.items()}
            imag = dyn.imagine_with_action(data["action"][:, 1:T], anchor)   # (B,T-1,...)
            fi = dyn.get_feat(imag)
            B = fi.shape[0]; H = T - 1
            # rewards/values
            r_p = wm.heads["reward"](fp[:, 1:T]).mean().squeeze(-1)          # (B,H)
            r_i = wm.heads["reward"](fi).mean().squeeze(-1)
            v_p = critic(fp[:, 1:T]).mean().squeeze(-1)
            v_i = critic(fi).mean().squeeze(-1)
            drift.append((r_i - r_p).abs().cpu().numpy().T + (v_i - v_p).abs().cpu().numpy().T)  # (H,B)
            latent.append((fi - fp[:, 1:T]).norm(dim=-1).cpu().numpy().T)
            spread.append(disc_spread(critic(fi)).cpu().numpy().T)
            pent.append(dyn.get_dist(imag).entropy().cpu().numpy().T)
            r_imag.append(r_i.cpu().numpy().T); v_imag.append(v_i.cpu().numpy().T)
            v_post.append(v_p.cpu().numpy().T); r_real.append(data["reward"][:, 1:T].cpu().numpy().T)
            # one-step KL(post||prior) per step as rho label, and prior feat for rho
            kld = torchd.kl.kl_divergence(dyn.get_dist(post), dyn.get_dist(prior))  # (B,T)
            kl_lab.append(kld[:, 1:T].cpu().numpy().T)
            rho_feat.append(dyn.get_feat(prior)[:, 1:T].cpu().numpy())               # (B,H,F)

    drift=np.concatenate(drift,1); latent=np.concatenate(latent,1)
    spread=np.concatenate(spread,1); pent=np.concatenate(pent,1)
    r_imag=np.concatenate(r_imag,1); v_imag=np.concatenate(v_imag,1)
    v_post=np.concatenate(v_post,1); r_real=np.concatenate(r_real,1)
    kl_lab=np.concatenate(kl_lab,1); rho_feat=np.concatenate(rho_feat,0)  # (B,H,F)
    H, N = drift.shape

    # ---- learned rho head: regress prior_feat -> one-step KL (train/eval split) ----
    Xtr = torch.tensor(rho_feat.reshape(-1, rho_feat.shape[-1]), dtype=torch.float32, device=DEV)
    ytr = torch.tensor(kl_lab.T.reshape(-1), dtype=torch.float32, device=DEV)
    rho = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ELU(), torch.nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.Adam(rho.parameters(), 1e-3)
    perm = torch.randperm(Xtr.shape[0]); half = Xtr.shape[0] // 2
    tr_i = perm[:half]
    for _ in range(800):
        bi = tr_i[torch.randint(0, half, (512,))]
        loss = torch.nn.functional.mse_loss(rho(Xtr[bi]).squeeze(-1), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sig_rho = rho(Xtr).squeeze(-1).cpu().numpy().reshape(N, H).T   # (H,N)

    # ---- signal validity ----
    res = {}
    for name, sig in [("critic_spread", spread), ("learned_rho", sig_rho), ("prior_entropy_BL08", pent)]:
        tc, per = t_controlled_spearman(sig, drift)
        res[name] = dict(t_controlled_spearman=float(tc), pooled_spearman=float(pooled_spearman(sig, drift)),
                         partial_spearman_ctrl_t=float(partial_spearman_controlling_t(sig, drift)),
                         per_horizon=[None if np.isnan(x) else float(x) for x in per])
    res["_latent_drift_check"] = {n: float(t_controlled_spearman(s, latent)[0]) for n, s in [("critic_spread", spread), ("learned_rho", sig_rho)]}

    # ---- feasibility: sign-unbiased selective bootstrap (pre-registered) ----
    gpow = GAMMA ** np.arange(H)
    G_truth = (gpow[:, None] * r_real).sum(0) + GAMMA ** H * v_post[H - 1]
    def T_of_m(m_arr):
        out = np.zeros(N)
        for n in range(N):
            m = int(m_arr[n]); out[n] = (gpow[:m] * r_imag[:m, n]).sum() + GAMMA ** m * v_imag[m - 1, n]
        return out
    def first_flag_m(sig, tau):
        m = np.full(N, H, int)
        for n in range(N):
            idx = np.where(sig[:, n] > tau)[0]
            if len(idx) > 0: m[n] = idx[0] + 1
        return m
    tune = np.arange(0, N // 2); test = np.arange(N // 2, N)
    def err_on(idx, Tv): return float(np.mean(np.abs(Tv[idx] - G_truth[idx])))
    T_full = T_of_m(np.full(N, H, int)); full_test = err_on(test, T_full); full_tune = err_on(tune, T_full)
    ag = {m: err_on(tune, T_of_m(np.full(N, m, int))) for m in range(1, H + 1)}
    ag_m = min(ag, key=ag.get); agnostic_test = err_on(test, T_of_m(np.full(N, ag_m, int)))
    def sigm(sig):
        qs = np.quantile(sig, np.linspace(0.05, 0.95, 19)); taus = sorted(set(list(qs) + [np.inf]))
        e = {float(t): err_on(tune, T_of_m(first_flag_m(sig, t))) for t in taus}
        bt = min(e, key=e.get)
        return dict(best_tau=float(bt), test_err=err_on(test, T_of_m(first_flag_m(sig, bt))), tune_err=float(e[bt]))
    feas = dict(full_test=full_test, full_tune=full_tune, agnostic_best_m=int(ag_m), agnostic_test=agnostic_test,
                oracle=sigm(drift), critic_spread=sigm(spread), learned_rho=sigm(sig_rho),
                sanity_full_recovered=bool(abs(full_tune - err_on(tune, T_of_m(first_flag_m(spread, np.inf)))) < 1e-6))
    res["feasibility_selective_bootstrap"] = feas

    # ---- verdict (same pre-registered criterion) ----
    GATE, MARGIN = 0.30, 0.95
    pr = {k: res[k]["t_controlled_spearman"] for k in ["critic_spread", "learned_rho"]}
    best = max(pr, key=lambda k: (pr[k] if not math.isnan(pr[k]) else -9))
    corr_pass = (not math.isnan(pr[best])) and pr[best] > GATE
    bar = MARGIN * min(feas["full_test"], feas["agnostic_test"])
    oracle_pass = feas["oracle"]["test_err"] <= bar
    learned_pass = feas["critic_spread"]["test_err"] <= bar
    if not feas["sanity_full_recovered"]:
        v, reason = "INVALID", "tau=inf did not recover full baseline"
    elif corr_pass and oracle_pass and learned_pass:
        v, reason = "CONDITIONAL-GO", "REAL DreamerV3: signal valid + selective-bootstrap (oracle AND learned) beats full & agnostic >=5%"
    elif corr_pass and oracle_pass:
        v, reason = "FAIL", "signal-quality: oracle helps but learned signal does not beat baselines"
    elif corr_pass:
        v, reason = "FAIL", "robust: even ORACLE does not beat full/agnostic -> mechanism no gain on REAL DreamerV3 (confirms synthetic)"
    else:
        v, reason = "FAIL", "signal validity gate failed on real model"
    res["_verdict"] = dict(verdict=v, reason=reason, best_signal=best, best_t_controlled_spearman=float(pr[best]),
        corr_pass=bool(corr_pass), oracle_pass=bool(oracle_pass), learned_pass=bool(learned_pass),
        bar_err=float(bar), full_test=float(feas["full_test"]), agnostic_test=float(feas["agnostic_test"]),
        oracle_test=float(feas["oracle"]["test_err"]), critic_spread_test=float(feas["critic_spread"]["test_err"]),
        N=int(N), H=int(H), logdir=a.logdir, note="Real DMControl (dmc_proprio walker_walk). Same pre-registered gate as synthetic.")
    with open(os.path.join(OUT, "experiment_real_results.json"), "w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res["_verdict"], indent=2))
    print("t-controlled Spearman:", {k: round(pr[k], 3) for k in pr}, "| BL08:", round(res["prior_entropy_BL08"]["t_controlled_spearman"], 3))
    print("feasibility:", json.dumps(feas, indent=2))

if __name__ == "__main__":
    main()
