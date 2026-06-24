"""
PoC R13 v3: embedding-only-task-onboarding
Tests on REAL TD-MPC2 mt30-1M checkpoint.

Key fixes from previous versions:
1. z_true uses FROZEN BASE ENCODER → prevents full FT from gaming metric via latent contraction
2. Q2 uses s2 (Mahalanobis distance of adapted e* from training emb hull) → independent predictor
3. Mini-batch training throughout
4. Gradient-through-target for embed-only (correct gradient when e enters both sides)

Hypothesis:
  Q1: at mass_scale=1.0 (in-manifold), embed-only gap < 20% of embed-only eval_loss
  Q2: Spearman(s2, gap) > 0.5 where s2 = L2 dist of adapted e* from nearest training embedding

Run from workspace root:
  MUJOCO_GL=egl LD_LIBRARY_PATH=/home/jovyan/egl_libs python \
    experiments/wip/embedding-only-task-onboarding/poc_r13.py
"""

import sys, os
sys.path.insert(0, '/home/jovyan/workspace/paper_agents_worldmodel/baselines/tdmpc2/tdmpc2')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LD_LIBRARY_PATH'] = '/home/jovyan/egl_libs:' + os.environ.get('LD_LIBRARY_PATH', '')

import copy
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = '/home/jovyan/workspace/paper_agents_worldmodel/baselines/tdmpc2/checkpoints/multitask/mt30-1M.pt'

MASS_SCALES = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
N_COLLECT = 2000
N_TRAIN = 1600
N_TEST = 400
ADAPT_STEPS = 500
BATCH = 256
LR_EMBED = 1e-3
LR_FULL  = 3e-4

HELD_OUT_ID = 30  # index in the 31-slot embedding table

np.random.seed(42)
torch.manual_seed(42)


# ── MinimalWM (encoder + dynamics + task_emb) ──────────────────────────────────
def _simnorm_cfg(d=8):
    return SimpleNamespace(simnorm_dim=d)

class MinimalWM(nn.Module):
    """
    Uses exact same layer functions as TD-MPC2 → state dict keys match checkpoint.
    Avoids Q-network TensorDictParams versioning issues.
    """
    def __init__(self, n_tasks=31, obs_dim=24, act_dim=6, task_dim=96,
                 latent_dim=128, enc_dim=256, mlp_dim=384, simnorm_dim=8):
        super().__init__()
        from common import layers as L
        sn_cfg = _simnorm_cfg(simnorm_dim)

        self._task_emb = nn.Embedding(n_tasks, task_dim, max_norm=1)
        enc_cfg = SimpleNamespace(
            obs_shape={'state': (obs_dim,)},
            task_dim=task_dim, num_enc_layers=2,
            enc_dim=enc_dim, latent_dim=latent_dim, simnorm_dim=simnorm_dim,
        )
        self._encoder = L.enc(enc_cfg)
        self._dynamics = L.mlp(
            latent_dim + act_dim + task_dim, 2*[mlp_dim], latent_dim,
            act=L.SimNorm(sn_cfg),
        )

    def encode(self, obs, task_id):
        if isinstance(task_id, int):
            task_id = torch.tensor([task_id], device=obs.device).expand(obs.shape[0])
        e = self._task_emb(task_id)
        return self._encoder['state'](torch.cat([obs, e], -1))

    def encode_with_embedding(self, obs, e):
        """Encode using an explicit embedding vector (bypasses task lookup)."""
        return self._encoder['state'](torch.cat([obs, e], -1))

    def next_latent(self, z, act, task_id):
        if isinstance(task_id, int):
            task_id = torch.tensor([task_id], device=z.device).expand(z.shape[0])
        e = self._task_emb(task_id)
        return self._dynamics(torch.cat([z, e, act], -1))


def load_model(device):
    """Load checkpoint, extend task_emb to 31 rows (walker-walk init for slot 30)."""
    from common import layers as L
    raw = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = dict(raw['model'])

    model = MinimalWM(n_tasks=31).to(device)
    emb_w = sd['_task_emb.weight']                              # [30, 96]
    init_row = emb_w[1:2].clone()                               # walker-walk
    full_emb = torch.cat([emb_w, init_row], dim=0).to(device)  # [31, 96]

    msd = model.state_dict()
    load_sd = {}
    for k in msd.keys():
        if k == '_task_emb.weight':
            load_sd[k] = full_emb
        elif k in sd:
            load_sd[k] = sd[k].to(device)
        else:
            load_sd[k] = msd[k]

    matched = [k for k in msd if k in sd or k == '_task_emb.weight']
    print(f'  Keys matched from checkpoint: {len(matched)}/{len(msd)}')
    model.load_state_dict(load_sd, strict=True)
    model.eval()
    print(f'  MinimalWM params: {sum(p.numel() for p in model.parameters()):,}')
    return model


# ── Environment / data collection ─────────────────────────────────────────────
def make_modified_walker(mass_scale):
    from dm_control import suite
    env = suite.load('walker', 'walk')
    for i in range(env.physics.model.nbody):
        env.physics.model.body_mass[i] *= mass_scale
    return env


def obs_flat(ts):
    return np.concatenate([v.flatten() for v in ts.observation.values()]).astype(np.float32)


def collect(mass_scale, n=N_COLLECT):
    env = make_modified_walker(mass_scale)
    ts = env.reset()
    s = obs_flat(ts)
    data = []
    for _ in range(n):
        a = np.random.uniform(-1, 1, 6).astype(np.float32)
        ts = env.step(a)
        s2 = obs_flat(ts)
        data.append((s.copy(), a, s2.copy(), float(ts.reward or 0.)))
        s = s2
        if ts.last():
            ts = env.reset(); s = obs_flat(ts)
    return data


def tensors(data, device):
    obs  = torch.tensor([d[0] for d in data], dtype=torch.float32, device=device)
    act  = torch.tensor([d[1] for d in data], dtype=torch.float32, device=device)
    nobs = torch.tensor([d[2] for d in data], dtype=torch.float32, device=device)
    return obs, act, nobs


# ── Loss functions ─────────────────────────────────────────────────────────────
def task_tensor(n, device):
    return torch.full((n,), HELD_OUT_ID, dtype=torch.long, device=device)


def wm_loss_embed_only(live_model, obs, act, nobs):
    """
    Embed-only: encoder frozen = base encoder.
    Gradient-through-target: both z_hat and z_true depend on e.
    Since encoder is frozen, no collapse risk.
    """
    task = task_tensor(obs.shape[0], obs.device)
    z     = live_model.encode(obs, task)
    z_hat = live_model.next_latent(z, act, task)
    z_true = live_model.encode(nobs, task)   # gradient flows (encoder frozen, only e changes)
    return (z_hat - z_true).pow(2).mean()


def wm_loss_full_ft(live_model, base_model, obs, act, nobs):
    """
    Full FT: use FROZEN BASE ENCODER for z_true to prevent latent contraction.
    z_hat: live (adapting) encoder + dynamics.
    z_true: base (frozen) encoder with live embedding.
    Gradient for encoder: only through z_hat (base enc in z_true has no grad).
    Gradient for embedding: through both (both use live e).
    """
    task = task_tensor(obs.shape[0], obs.device)
    # z_hat: live model (encoder and dynamics may adapt)
    z     = live_model.encode(obs, task)
    z_hat = live_model.next_latent(z, act, task)
    # z_true: base encoder (frozen) with live embedding
    e = live_model._task_emb(task)                     # gradient flows through e
    z_true = base_model.encode_with_embedding(nobs, e) # base_model weights have no grad
    return (z_hat - z_true).pow(2).mean()


@torch.no_grad()
def eval_loss_collapsed(live_model, base_model, obs, act, nobs):
    """
    Collapse-proof eval: z_true always from BASE ENCODER + LIVE EMBEDDING.
    Both embed-only and full FT are scored in the SAME latent geometry.
    """
    task = task_tensor(obs.shape[0], obs.device)
    z     = live_model.encode(obs, task)
    z_hat = live_model.next_latent(z, act, task)
    e = live_model._task_emb(task)
    z_true = base_model.encode_with_embedding(nobs, e)
    return (z_hat - z_true).pow(2).mean().item()


# ── Adaptation ─────────────────────────────────────────────────────────────────
def adapt_embed_only(base_model, obs_tr, act_tr, nobs_tr,
                     steps=ADAPT_STEPS, lr=LR_EMBED):
    """Freeze backbone, optimize only held-out embedding row (96 params)."""
    model = copy.deepcopy(base_model)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    model._task_emb.weight.requires_grad_(True)
    optim = torch.optim.Adam([model._task_emb.weight], lr=lr)

    for _ in range(steps):
        optim.zero_grad()
        idx = torch.randint(0, obs_tr.shape[0], (BATCH,), device=obs_tr.device)
        loss = wm_loss_embed_only(model, obs_tr[idx], act_tr[idx], nobs_tr[idx])
        loss.backward()
        # Mask: only held-out row gets gradient update
        if model._task_emb.weight.grad is not None:
            model._task_emb.weight.grad[:HELD_OUT_ID].zero_()
            model._task_emb.weight.grad[HELD_OUT_ID+1:].zero_()
        optim.step()

    model.eval()
    return model


def adapt_full_ft(base_model, obs_tr, act_tr, nobs_tr,
                  steps=ADAPT_STEPS, lr=LR_FULL):
    """Optimize all parameters; z_true anchored to base encoder to prevent collapse."""
    model = copy.deepcopy(base_model)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optim.zero_grad()
        idx = torch.randint(0, obs_tr.shape[0], (BATCH,), device=obs_tr.device)
        loss = wm_loss_full_ft(model, base_model, obs_tr[idx], act_tr[idx], nobs_tr[idx])
        loss.backward()
        optim.step()

    model.eval()
    return model


# ── s2: embedding-space distance ───────────────────────────────────────────────
@torch.no_grad()
def compute_s2(adapted_model, base_model):
    """
    s2 = min L2 distance of adapted e* from any of the 30 pretrained embeddings.
    Independent predictor: computed in embedding space, not from loss.
    """
    e_star = adapted_model._task_emb.weight[HELD_OUT_ID]   # [96]
    E_train = base_model._task_emb.weight[:30]             # [30, 96]
    dists = (E_train - e_star.unsqueeze(0)).pow(2).sum(-1)  # [30]
    return dists.min().item()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    print('Loading mt30-1M checkpoint...')
    base_model = load_model(DEVICE)
    # base_model is ALWAYS frozen (used as reference target)
    for p in base_model.parameters():
        p.requires_grad_(False)
    print()

    results = []
    for ms in MASS_SCALES:
        print(f'── mass_scale={ms:.1f} ──')

        data = collect(ms, n=N_COLLECT)
        obs_tr, act_tr, nobs_tr = tensors(data[:N_TRAIN], DEVICE)
        obs_te, act_te, nobs_te = tensors(data[N_TRAIN:N_TRAIN+N_TEST], DEVICE)

        # Baseline: no adaptation
        loss_base = eval_loss_collapsed(base_model, base_model, obs_te, act_te, nobs_te)

        # A: embed-only
        model_a = adapt_embed_only(base_model, obs_tr, act_tr, nobs_tr)
        loss_a  = eval_loss_collapsed(model_a, base_model, obs_te, act_te, nobs_te)
        s2_a    = compute_s2(model_a, base_model)

        # B: full FT
        model_b = adapt_full_ft(base_model, obs_tr, act_tr, nobs_tr)
        loss_b  = eval_loss_collapsed(model_b, base_model, obs_te, act_te, nobs_te)

        gap = loss_a - loss_b
        print(f'  base  : {loss_base:.6f}')
        print(f'  embed : {loss_a:.6f}  s2={s2_a:.4f}')
        print(f'  fullFT: {loss_b:.6f}')
        print(f'  gap   : {gap:.6f}')
        print()

        results.append({'ms': ms, 's2': s2_a, 'gap': gap,
                        'loss_base': loss_base, 'loss_a': loss_a, 'loss_b': loss_b})

    # ── Gate checks ───────────────────────────────────────────────────────────
    s2_vals  = [r['s2']  for r in results]
    gap_vals = [r['gap'] for r in results]

    # Q1: at mass_scale=1.0, gap < 20% of embed-only loss
    idx_1 = MASS_SCALES.index(1.0)
    r1 = results[idx_1]
    q1_thresh = 0.20 * r1['loss_a']
    q1_pass = r1['gap'] < q1_thresh
    print(f'Q1 (mass=1.0 gap < 20% embed-only): gap={r1["gap"]:.6f}, thresh={q1_thresh:.6f} → {"PASS ✓" if q1_pass else "FAIL ✗"}')

    # Q2: Spearman(s2, gap) > 0.5
    rho, pval = stats.spearmanr(s2_vals, gap_vals)
    q2_pass = rho > 0.5
    print(f'Q2 Spearman(s2, gap): rho={rho:.3f}, p={pval:.4f} → {"PASS ✓" if q2_pass else "FAIL ✗"}')

    poc_pass = q1_pass and q2_pass
    print()
    print(f'═══ PoC result: {"PASS ✓" if poc_pass else "FAIL ✗"} ═══')
    print()
    print(f'{"mass":>6}  {"base":>10}  {"embed":>10}  {"fullFT":>10}  {"gap":>10}  {"s2":>8}')
    for r in results:
        print(f'{r["ms"]:>6.1f}  {r["loss_base"]:>10.6f}  {r["loss_a"]:>10.6f}  '
              f'{r["loss_b"]:>10.6f}  {r["gap"]:>10.6f}  {r["s2"]:>8.4f}')

    return poc_pass, results, rho


if __name__ == '__main__':
    poc_pass, results, rho = main()
    import sys
    sys.exit(0 if poc_pass else 1)
