"""
DIE-POINT microbenchmark for elite-staged-value-planning (advisor-ordered FIRST).

Question: does the proposed 2-stage terminal-value pattern make TD-MPC2 _plan
faster in WALL-CLOCK (not FLOPs), on GPU AND CPU?

Proposal adds a 512-sample cheap-rank forward + does full ensemble Q on only 64
elites, vs baseline's full ensemble Q on all 512. On B200 the Q-ensemble is a
single vectorized (vmap) op over tiny MLPs -> reducing 512->64 is a batch-dim
change on a kernel-launch-bound op, and we ADD a cheap-rank launch. So GPU may be
flat/negative; CPU (FLOP-bound, the real-robot/edge regime) may show the gain.

cheap-rank = a single distilled scalar MLP (BEST CASE for the idea: cheapest
possible cheap-rank). If even this doesn't beat baseline >=1.2x on some device,
the idea is dead there. Elite selection is STUBBED (random 64) -- timing only.

PRE-REGISTERED gate: proposed >= 1.2x faster on per-env-step _plan wall-clock,
on GPU or CPU. If GPU<1.2x but CPU>=1.2x -> pivot claim to edge/CPU deployment.
If both <1.2x -> FAIL.
"""
import os, sys, time, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
TM = "/home/jovyan/workspace/paper_agents_worldmodel/baselines/tdmpc2/tdmpc2"
sys.path.insert(0, TM)
import numpy as np, torch, torch.nn as nn
from omegaconf import OmegaConf
import hydra
from common.parser import parse_cfg
from common import math as tdmath
from envs import make_env
from tdmpc2 import TDMPC2
OUT = os.path.dirname(os.path.abspath(__file__))

def build(device):
    import hydra.utils as _hu  # parse_cfg calls get_original_cwd(); stub it (no hydra runtime)
    _hu.get_original_cwd = lambda: TM
    cfg = OmegaConf.load(os.path.join(TM, "config.yaml"))
    cfg.task = "walker-walk"; cfg.obs = "state"; cfg.model_size = 5
    cfg.steps = 10000; cfg.seed_steps = 500; cfg.seed = 0
    cfg.enable_wandb = False; cfg.wandb_project = "x"; cfg.wandb_entity = "x"
    cfg.compile = False; cfg.work_dir = "/data/jameskimh/worldmodel/tdmpc2-microbench"
    cfg.multitask = False
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    cfg.device = device
    agent = TDMPC2(cfg)
    agent.model = agent.model.to(device)          # TD-MPC2 hardcodes cuda; force target device
    if torch.is_tensor(agent.discount):
        agent.discount = agent.discount.to(device)
    agent.model.eval()
    return cfg, env, agent

class CheapHead(nn.Module):
    """distilled scalar value head (best-case cheap-rank): latent+action -> 1."""
    def __init__(self, latent_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim+action_dim, hidden), nn.Mish(),
                                 nn.Linear(hidden, 1))
    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)

@torch.no_grad()
def plan_baseline(agent, z0):
    """Replica of TD-MPC2 _plan (single env step), baseline terminal (qnet on all 512)."""
    cfg = agent.cfg; m = agent.model; task = None
    H, N, NE, NP = cfg.horizon, cfg.num_samples, cfg.num_elites, cfg.num_pi_trajs
    z = z0.repeat(N, 1)
    mean = torch.zeros(H, cfg.action_dim, device=cfg.device)
    std = torch.full((H, cfg.action_dim), cfg.max_std, device=cfg.device)
    actions = torch.empty(H, N, cfg.action_dim, device=cfg.device)
    for _ in range(cfg.iterations):
        r = torch.randn(H, N, cfg.action_dim, device=cfg.device)
        actions = (mean.unsqueeze(1) + std.unsqueeze(1)*r).clamp(-1,1)
        # _estimate_value: dynamics term (512) + terminal (qnet avg on 512)
        zt = z; G = 0; disc = 1
        for t in range(H):
            rew = tdmath.two_hot_inv(m.reward(zt, actions[t], task), cfg)
            zt = m.next(zt, actions[t], task)
            G = G + disc*rew; disc = disc*agent.discount
        a_term, _ = m.pi(zt, task)
        value = (G + disc*m.Q(zt, a_term, task, return_type='avg')).nan_to_num(0)
        elite = torch.topk(value.squeeze(1), NE, dim=0).indices
        ev, ea = value[elite], actions[:, elite]
        mv = ev.max(0).values
        score = torch.exp(cfg.temperature*(ev-mv)); score = score/score.sum(0)
        mean = (score.unsqueeze(0)*ea).sum(1)/(score.sum(0)+1e-9)
        std = ((score.unsqueeze(0)*(ea-mean.unsqueeze(1))**2).sum(1)/(score.sum(0)+1e-9)).sqrt().clamp(cfg.min_std,cfg.max_std)
    return mean[0]

@torch.no_grad()
def plan_2stage(agent, z0, cheap):
    """2-stage: cheap-rank (scalar head) on 512 -> select 64 -> full qnet on 64 only."""
    cfg = agent.cfg; m = agent.model; task = None
    H, N, NE, NP = cfg.horizon, cfg.num_samples, cfg.num_elites, cfg.num_pi_trajs
    z = z0.repeat(N, 1)
    mean = torch.zeros(H, cfg.action_dim, device=cfg.device)
    std = torch.full((H, cfg.action_dim), cfg.max_std, device=cfg.device)
    for _ in range(cfg.iterations):
        r = torch.randn(H, N, cfg.action_dim, device=cfg.device)
        actions = (mean.unsqueeze(1) + std.unsqueeze(1)*r).clamp(-1,1)
        # dynamics term on all 512 (same as baseline)
        zt = z; G = 0; disc = 1
        for t in range(H):
            rew = tdmath.two_hot_inv(m.reward(zt, actions[t], task), cfg)
            zt = m.next(zt, actions[t], task)
            G = G + disc*rew; disc = disc*agent.discount
        a_term, _ = m.pi(zt, task)
        # STAGE 1: cheap-rank on all 512 -> elite 64
        v_cheap = cheap(zt, a_term)                                   # (512,) single MLP
        cheap_total = G.squeeze(1) + disc*v_cheap
        elite = torch.topk(cheap_total, NE, dim=0).indices            # 64
        # STAGE 2: full ensemble qnet on 64 elites only
        v_full = m.Q(zt[elite], a_term[elite], task, return_type='avg')  # (64,1) vectorized all-5
        ev = G[elite] + disc*v_full
        ea = actions[:, elite]
        mv = ev.max(0).values
        score = torch.exp(cfg.temperature*(ev-mv)); score = score/score.sum(0)
        mean = (score.unsqueeze(0)*ea).sum(1)/(score.sum(0)+1e-9)
        std = ((score.unsqueeze(0)*(ea-mean.unsqueeze(1))**2).sum(1)/(score.sum(0)+1e-9)).sqrt().clamp(cfg.min_std,cfg.max_std)
    return mean[0]

def timeit(fn, n_warmup, n_iter, device):
    for _ in range(n_warmup): fn()
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n_iter*1000  # ms/step

def run(device):
    torch.manual_seed(0); np.random.seed(0)
    cfg, env, agent = build(device)
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    obs = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
    z0 = agent.model.encode(obs, None)
    cheap = CheapHead(cfg.latent_dim, cfg.action_dim).to(device).eval()
    nW, nI = 5, 30
    base = timeit(lambda: plan_baseline(agent, z0), nW, nI, device)
    two = timeit(lambda: plan_2stage(agent, z0, cheap), nW, nI, device)
    return dict(device=device, baseline_ms=round(base,3), twostage_ms=round(two,3),
                speedup=round(base/two,3))

def main():
    res = {}
    for key, dev in [("gpu", "cuda:0"), ("cpu", "cpu")]:
        try:
            res[key] = run(dev)
            print(f"[{key}] {res[key]}", flush=True)
        except Exception as e:
            res[key] = dict(device=dev, error=repr(e))
            print(f"[{key}] ERROR {e!r}", flush=True)
    GATE = 1.2
    gpu_pass = res["gpu"].get("speedup", 0) >= GATE
    cpu_pass = res["cpu"].get("speedup", 0) >= GATE
    if gpu_pass:
        verdict, claim = "PASS", "GPU wall-clock speedup -- general"
    elif cpu_pass:
        verdict, claim = "PASS-CPU-ONLY", "GPU flat; CPU/edge speedup -> pivot claim to resource-constrained deployment"
    else:
        verdict, claim = "FAIL", "neither GPU nor CPU >=1.2x plan wall-clock -> 2-stage adds launches, vectorized qnet not the wall-clock bottleneck"
    res["_verdict"] = dict(verdict=verdict, claim=claim, gate=GATE,
                           gpu_speedup=res["gpu"]["speedup"], cpu_speedup=res["cpu"]["speedup"],
                           note="cheap-rank=distilled scalar MLP (best case). elite stubbed (timing only). config: 512 samples, 6 iter, H3, 64 elites, num_q5(avg=2 subsampled), latent512.")
    with open(os.path.join(OUT, "microbench_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
