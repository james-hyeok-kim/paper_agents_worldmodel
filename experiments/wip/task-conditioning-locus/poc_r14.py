"""
PoC R14 v3: task-conditioning-locus
Tests which module of TD-MPC2 is sensitive to wrong task embedding.

Hypothesis:
  Within same-embodiment family (walker-{stand,walk,run}):
  - DYNAMICS prediction is ROBUST to wrong task embedding (sensitivity low)
  - REWARD prediction is FRAGILE to wrong task embedding (sensitivity high)
  → sensitivity_ratio(reward/dynamics) > 3

v3 changes from v2:
  - DYNAMICS: random rollout data (correct — no reward variation needed)
  - REWARD: velocity sweep on upright states (fixes v2 confound where random
    data had no task-discriminating velocity variation)
    * qvel[0] swept over [0, 0.5, 1, 2, 4, 8, 12] m/s on upright base states
    * stand reward is HIGH at v=0; walk HIGH at v=1; run HIGH at v=8
  - Gates unchanged: Q1=mean(rew/dyn ratio)>3, Q2=mean(dyn_sens)<0.5

Run:
  MUJOCO_GL=egl LD_LIBRARY_PATH=/home/jovyan/egl_libs python \
    experiments/wip/task-conditioning-locus/poc_r14.py
"""

import sys, os
sys.path.insert(0, '/home/jovyan/workspace/paper_agents_worldmodel/baselines/tdmpc2/tdmpc2')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LD_LIBRARY_PATH'] = '/home/jovyan/egl_libs:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = '/home/jovyan/workspace/paper_agents_worldmodel/baselines/tdmpc2/checkpoints/multitask/mt30-1M.pt'

N_COLLECT    = 1000   # random steps for dynamics test
N_BASE_STATES = 5     # upright states for velocity sweep
VMIN, VMAX   = -10.0, 10.0
NUM_BINS     = 101
VELOCITY_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

np.random.seed(42)
torch.manual_seed(42)

TASK_IDS = {
    'walker-stand': 0,
    'walker-walk':  1,
    'walker-run':   2,
}
SAME_FAMILY_PAIRS = [
    ('walker-stand', 'walker-walk'),
    ('walker-stand', 'walker-run'),
    ('walker-walk',  'walker-run'),
]


# ── Model ─────────────────────────────────────────────────────────────────────
class MinimalWMWithReward(nn.Module):
    """Encoder + Dynamics + Reward + task_emb. State-dict keys match checkpoint."""
    def __init__(self, n_tasks=30, obs_dim=24, act_dim=6, task_dim=96,
                 latent_dim=128, enc_dim=256, mlp_dim=384, simnorm_dim=8,
                 num_bins=101):
        super().__init__()
        from common import layers as L
        sn_cfg = SimpleNamespace(simnorm_dim=simnorm_dim)
        self._task_emb = nn.Embedding(n_tasks, task_dim, max_norm=1)
        enc_cfg = SimpleNamespace(
            obs_shape={'state': (obs_dim,)},
            task_dim=task_dim, num_enc_layers=2,
            enc_dim=enc_dim, latent_dim=latent_dim, simnorm_dim=simnorm_dim,
        )
        self._encoder  = L.enc(enc_cfg)
        self._dynamics = L.mlp(
            latent_dim + act_dim + task_dim, 2*[mlp_dim], latent_dim,
            act=L.SimNorm(sn_cfg),
        )
        self._reward = L.mlp(latent_dim + act_dim + task_dim, 2*[mlp_dim], num_bins)

    def get_emb(self, task_id: int, batch_size: int):
        tid = torch.full((batch_size,), task_id, dtype=torch.long, device=DEVICE)
        return self._task_emb(tid)

    def encode(self, obs, e):
        return self._encoder['state'](torch.cat([obs, e], -1))

    def pred_next_latent(self, z, act, e):
        return self._dynamics(torch.cat([z, act, e], -1))

    def pred_reward(self, z, act, e):
        return self._reward(torch.cat([z, act, e], -1))  # [B, num_bins]


def load_model():
    raw = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd  = dict(raw['model'])
    model = MinimalWMWithReward(n_tasks=30).to(DEVICE)
    msd = model.state_dict()
    load_sd = {}
    for k in msd:
        if k in sd:
            load_sd[k] = sd[k].to(DEVICE)
        else:
            load_sd[k] = msd[k]
            print(f'  WARNING: {k} missing from ckpt, using random init')
    model.load_state_dict(load_sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'  Keys matched: {sum(1 for k in msd if k in sd)}/{len(msd)}')
    return model


# ── Environment helpers ────────────────────────────────────────────────────────
def make_env(task_name):
    from dm_control import suite
    name_map = {
        'walker-stand': ('walker', 'stand'),
        'walker-walk':  ('walker', 'walk'),
        'walker-run':   ('walker', 'run'),
    }
    domain, task = name_map[task_name]
    return suite.load(domain, task)


def obs_flat_ts(ts):
    return np.concatenate([v.flatten() for v in ts.observation.values()]).astype(np.float32)


def obs_flat_dict(obs_dict):
    return np.concatenate([v.flatten() for v in obs_dict.values()]).astype(np.float32)


# ── Data collection ────────────────────────────────────────────────────────────
def collect_random(task_name, n=N_COLLECT):
    """Random rollout — used for dynamics sensitivity test."""
    env = make_env(task_name)
    ts  = env.reset()
    s   = obs_flat_ts(ts)
    data = {'obs': [], 'act': [], 'nobs': [], 'rew': []}
    for _ in range(n):
        a  = np.random.uniform(-1, 1, 6).astype(np.float32)
        ts = env.step(a)
        s2 = obs_flat_ts(ts)
        data['obs'].append(s.copy())
        data['act'].append(a)
        data['nobs'].append(s2.copy())
        data['rew'].append(float(ts.reward or 0.))
        s = s2
        if ts.last():
            ts = env.reset(); s = obs_flat_ts(ts)
    return {k: torch.tensor(np.array(v), dtype=torch.float32, device=DEVICE)
            for k, v in data.items()}


def _stand_reward_proxy(env_stand, qpos, qvel):
    """Compute walker-stand reward as height-proxy (task-agnostic upright criterion)."""
    env_stand.physics.data.qpos[:] = qpos.copy()
    env_stand.physics.data.qvel[:] = qvel.copy()
    env_stand.physics.forward()
    return float(env_stand.task.get_reward(env_stand.physics))


def create_velocity_sweep(task_name):
    """
    Sweep qvel[1] (forward X-velocity) over VELOCITY_SWEEP on upright base states.
    Ground-truth reward from the task's own reward function.
    stand reward peaks at v=0; walk at v≈1; run at v≈8 → task-discriminating.

    Upright criterion uses walker-stand reward as height proxy (task-agnostic),
    so walker-walk/run also collect N_BASE_STATES upright states.
    """
    env       = make_env(task_name)
    env_stand = make_env('walker-stand')

    # Always include reset state
    ts = env.reset()
    env.physics.forward()
    base_states = [(env.physics.data.qpos.copy(), env.physics.data.qvel.copy())]

    for _ in range(4000):
        a  = np.random.uniform(-0.5, 0.5, 6).astype(np.float32)
        ts = env.step(a)
        stand_r = _stand_reward_proxy(env_stand,
                                       env.physics.data.qpos,
                                       env.physics.data.qvel)
        if stand_r > 0.6:
            base_states.append((env.physics.data.qpos.copy(), env.physics.data.qvel.copy()))
        if len(base_states) >= N_BASE_STATES:
            break
        if ts.last():
            ts = env.reset()

    print(f'    {task_name}: {len(base_states)} upright base states')

    obs_list, act_list, rew_list = [], [], []
    for qpos, qvel_base in base_states:
        for vel in VELOCITY_SWEEP:
            env.physics.data.qpos[:] = qpos
            qvel_new    = qvel_base.copy()
            qvel_new[1] = vel   # qvel[1] = forward X-velocity (qvel[0] is vertical Z)
            env.physics.data.qvel[:] = qvel_new
            env.physics.forward()                      # propagate kinematics

            obs = obs_flat_dict(env.task.get_observation(env.physics))
            rew = float(env.task.get_reward(env.physics))

            obs_list.append(obs)
            act_list.append(np.zeros(6, dtype=np.float32))
            rew_list.append(rew)

    rews = rew_list
    print(f'    {task_name}: {len(obs_list)} states, rew=[{min(rews):.3f}…{max(rews):.3f}]')

    return {
        'obs': torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE),
        'act': torch.tensor(np.array(act_list), dtype=torch.float32, device=DEVICE),
        'rew': torch.tensor(np.array(rew_list), dtype=torch.float32, device=DEVICE),
    }


# ── Loss functions ─────────────────────────────────────────────────────────────
_BINS = None

def get_bins(device):
    global _BINS
    if _BINS is None or _BINS.device != device:
        _BINS = torch.linspace(VMIN, VMAX, NUM_BINS, device=device)
    return _BINS


@torch.no_grad()
def dynamics_loss(model, data, e_enc, e_dyn):
    """
    Isolates dynamics head: encoder uses e_enc (fixed), dynamics uses e_dyn (variable).
    """
    B     = data['obs'].shape[0]
    e_enc_ = e_enc.expand(B, -1)
    e_dyn_ = e_dyn.expand(B, -1)
    z      = model.encode(data['obs'],  e_enc_)
    z_hat  = model.pred_next_latent(z, data['act'], e_dyn_)
    z_true = model.encode(data['nobs'], e_enc_)
    return (z_hat - z_true).pow(2).mean().item()


@torch.no_grad()
def reward_loss(model, data, e_enc, e_rew):
    """
    Isolates reward head: encoder uses e_enc (fixed), reward uses e_rew (variable).
    MSE of predicted mean reward vs actual reward from velocity-sweep data.
    """
    B     = data['obs'].shape[0]
    e_enc_ = e_enc.expand(B, -1)
    e_rew_ = e_rew.expand(B, -1)
    z      = model.encode(data['obs'], e_enc_)
    logits = model.pred_reward(z, data['act'], e_rew_)   # [B, 101]
    bins   = get_bins(logits.device)
    probs  = torch.softmax(logits, dim=-1)
    r_pred = (probs * bins.unsqueeze(0)).sum(-1)          # [B]
    return (r_pred - data['rew']).pow(2).mean().item()


def sensitivity(loss_wrong, loss_correct):
    return (loss_wrong - loss_correct) / (abs(loss_correct) + 1e-8)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    print('Loading checkpoint...')
    model = load_model()
    print()

    print('Collecting random rollout data (dynamics test)...')
    rnd_data = {}
    for task in TASK_IDS:
        print(f'  {task}...')
        rnd_data[task] = collect_random(task)
    print()

    print('Creating velocity sweep data (reward test)...')
    vel_data = {}
    for task in TASK_IDS:
        vel_data[task] = create_velocity_sweep(task)
    print()

    # ── Baseline (correct embedding everywhere) ────────────────────────────
    print('=== BASELINE (correct embedding) ===')
    for task, tid in TASK_IDS.items():
        e = model.get_emb(tid, 1).squeeze(0)
        ld = dynamics_loss(model, rnd_data[task], e, e)
        lr = reward_loss(model,   vel_data[task], e, e)
        print(f'  {task}: dyn_loss={ld:.5f}  rew_loss={lr:.5f}')
    print()

    # ── Same-family swap sensitivity ───────────────────────────────────────
    print('=== SAME-FAMILY SWAP SENSITIVITY ===')
    results = []
    for src_task, swap_task in SAME_FAMILY_PAIRS:
        src_id  = TASK_IDS[src_task]
        swap_id = TASK_IDS[swap_task]
        e_src  = model.get_emb(src_id,  1).squeeze(0)
        e_swap = model.get_emb(swap_id, 1).squeeze(0)

        ld_correct = dynamics_loss(model, rnd_data[src_task], e_src,  e_src)
        lr_correct = reward_loss(  model, vel_data[src_task], e_src,  e_src)
        ld_wrong   = dynamics_loss(model, rnd_data[src_task], e_src,  e_swap)
        lr_wrong   = reward_loss(  model, vel_data[src_task], e_src,  e_swap)

        s_dyn = sensitivity(ld_wrong, ld_correct)
        s_rew = sensitivity(lr_wrong, lr_correct)
        ratio = s_rew / (abs(s_dyn) + 1e-8)

        print(f'  {src_task} → swap {swap_task}:')
        print(f'    dyn:  correct={ld_correct:.5f} wrong={ld_wrong:.5f} Δ={s_dyn:+.3f}')
        print(f'    rew:  correct={lr_correct:.5f} wrong={lr_wrong:.5f} Δ={s_rew:+.3f}')
        print(f'    ratio(rew_Δ/dyn_Δ) = {ratio:.2f}')
        results.append({'src': src_task, 'swap': swap_task,
                        's_dyn': s_dyn, 's_rew': s_rew, 'ratio': ratio,
                        'ld_c': ld_correct, 'ld_w': ld_wrong,
                        'lr_c': lr_correct, 'lr_w': lr_wrong})
    print()

    # ── Gate checks ────────────────────────────────────────────────────────
    mean_ratio   = np.mean([r['ratio'] for r in results])
    mean_dyn_sns = np.mean([r['s_dyn'] for r in results])

    q1_pass = mean_ratio > 3.0
    q2_pass = mean_dyn_sns < 0.5

    print(f'Q1 mean(rew_Δ/dyn_Δ) > 3:     {mean_ratio:.2f}  → {"PASS ✓" if q1_pass else "FAIL ✗"}')
    print(f'Q2 mean(dyn_sensitivity) < 0.5: {mean_dyn_sns:.3f} → {"PASS ✓" if q2_pass else "FAIL ✗"}')
    poc_pass = q1_pass and q2_pass
    print()
    print(f'═══ PoC result: {"PASS ✓" if poc_pass else "FAIL ✗"} ═══')

    return poc_pass, results


if __name__ == '__main__':
    poc_pass, results = main()
    sys.exit(0 if poc_pass else 1)
