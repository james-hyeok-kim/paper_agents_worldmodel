"""
K-chunked imagination loop 테스트
DualRateRSSM의 핵심 문제: Python-level sequential GRU call overhead

해결책: K step을 하나의 청크로 묶어 처리
- step 0: fast + slow GRU  (slow update)
- step 1~K-1: fast GRU only (slow carry-forward)
이 패턴을 반복 → Python loop iteration이 K배 줄어듦
"""
import os, sys, time, json
import pathlib
import numpy as np

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, str(pathlib.Path(__file__).parent / "dreamerv3-torch"))

import torch
import tools, yaml, networks, models
import gymnasium as _gym, gymnasium.spaces as _gym_spaces, types as _types
_gym_mod = _types.ModuleType('gym')
_gym_mod.spaces = _gym_spaces
_gym_mod.Wrapper = _gym.Wrapper
sys.modules['gym'] = _gym_mod

import argparse
import collections


def load_config_for(config_names, base_dir):
    configs = yaml.safe_load(pathlib.Path(base_dir / "configs.yaml").read_text())
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    defaults = {}
    for name in config_names:
        if name in configs:
            recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    c = parser.parse_args([])
    c.device = 'cuda:0'
    c.num_actions = 6
    c.compile = False
    return c


def build_initial_state(dynamics, batch_size, device):
    if isinstance(dynamics, networks.DualRateRSSM):
        deter_total = dynamics._deter_total
        state = dict(
            logit=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            stoch=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            deter=torch.zeros(batch_size, deter_total, device=device),
            step_idx=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
        )
    else:
        state = dict(
            logit=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            stoch=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            deter=torch.zeros(batch_size, dynamics._deter, device=device),
        )
    return state


def rollout_standard(dynamics, state, action, horizon):
    """Standard rollout: one img_step() call per horizon step."""
    s = {k: v.clone() for k, v in state.items()}
    for _ in range(horizon):
        s = dynamics.img_step(s, action)
    return s


def rollout_chunked(dynamics, state, action, horizon, K):
    """
    K-chunked rollout: group into K-sized chunks.
    Within each chunk: 1 slow update + (K-1) fast-only updates.
    Reduces Python iterations by K (from horizon to horizon/K outer loops).
    Note: still calls img_step internally, but could be further optimized.
    """
    assert isinstance(dynamics, networks.DualRateRSSM), "Only for DualRateRSSM"

    s = {k: v.clone() for k, v in state.items()}
    n_chunks = (horizon + K - 1) // K  # ceil div
    steps_done = 0

    for chunk_i in range(n_chunks):
        chunk_steps = min(K, horizon - steps_done)
        for local_step in range(chunk_steps):
            s = dynamics.img_step(s, action)
        steps_done += chunk_steps

    return s


def rollout_fused(dynamics, state, action, horizon, K):
    """
    Fused K-chunked rollout: directly calls fast/slow GRU without img_step overhead.
    - slow update: every K steps (1x ib_proj + slow_inp + cell_slow)
    - fast update: every step (1x fast_inp + cell_fast)
    - stoch sampling: every step
    This avoids repeated dict creation and scalar branch check.
    """
    assert isinstance(dynamics, networks.DualRateRSSM)

    prev_stoch = state['stoch']
    prev_deter = state['deter']
    step_idx = state.get('step_idx', torch.zeros(prev_stoch.shape[0], 1, dtype=torch.long, device=prev_stoch.device))

    deter_slow = prev_deter[:, :dynamics._deter_slow]
    deter_fast = prev_deter[:, dynamics._deter_slow:]

    if dynamics._discrete:
        stoch_shape = list(prev_stoch.shape[:-2]) + [dynamics._stoch * dynamics._discrete]
        stoch_flat = prev_stoch.reshape(stoch_shape)
    else:
        stoch_flat = prev_stoch

    # Pre-compute base step_idx
    base_step = step_idx.squeeze(-1)[0].item()  # scalar (uniform batch)

    for h in range(horizon):
        global_step = int(base_step) + h

        # Fast update (every step)
        x_fast = torch.cat([stoch_flat, action, deter_slow], -1)
        x_fast = dynamics._fast_inp_layers(x_fast)
        x_fast, new_deter_fast_list = dynamics._cell_fast(x_fast, [deter_fast])
        deter_fast = new_deter_fast_list[0]

        # Slow update (every K steps only)
        if global_step % dynamics._slow_K == 0:
            fast_for_slow = dynamics._ib_proj(deter_fast)
            x_slow = dynamics._slow_inp_layers(fast_for_slow)
            x_slow, new_deter_slow_list = dynamics._cell_slow(x_slow, [deter_slow])
            deter_slow = new_deter_slow_list[0]
        # else: deter_slow carries forward unchanged

        # Stoch sampling
        new_deter = torch.cat([deter_slow, deter_fast], -1)
        x = dynamics._img_out_layers(new_deter)
        stats = dynamics._suff_stats_layer("ims", x)
        stoch = dynamics.get_dist(stats).sample()

        if dynamics._discrete:
            stoch_flat = stoch.reshape(list(stoch.shape[:-2]) + [dynamics._stoch * dynamics._discrete])
        else:
            stoch_flat = stoch

    new_step_idx = (step_idx + horizon)
    return {
        'stoch': stoch,
        'deter': torch.cat([deter_slow, deter_fast], -1),
        'step_idx': new_step_idx,
        **stats,
    }


def benchmark_rollout_fn(fn, state, action, horizon, n_trials=30, label=""):
    # warmup
    for _ in range(5):
        fn(state, action, horizon)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(state, action, horizon)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    batch_size = action.shape[0]
    mean_lat = float(np.mean(latencies))
    sps = (batch_size * horizon) / mean_lat
    print(f"  {label}: {sps:.0f} steps/sec, {mean_lat*1000:.2f}ms")
    return sps, mean_lat


def main():
    base_dir = pathlib.Path("dreamerv3-torch")
    device = torch.device('cuda:0')

    ObsSpace = collections.namedtuple('ObsSpace', ['spaces'])
    Box = collections.namedtuple('Box', ['shape'])
    obs_space = ObsSpace(spaces={'image': Box(shape=(64, 64, 3))})

    class FakeStep:
        value = 0
        def __mul__(self, other): return 0

    baseline_config = load_config_for(['defaults', 'dmc_vision'], base_dir)
    baseline_wm = models.WorldModel(obs_space, None, FakeStep(), baseline_config).to(device)
    baseline_wm.eval()

    dualrate_config = load_config_for(['defaults', 'dmc_vision', 'dual_rate'], base_dir)
    dualrate_wm = models.WorldModel(obs_space, None, FakeStep(), dualrate_config).to(device)
    dualrate_wm.eval()

    K = dualrate_config.slow_K  # 3
    horizon = 15

    all_results = {}

    for batch_size in [512, 1024, 4096]:
        print(f"\n{'='*50}")
        print(f"Batch size = {batch_size}, horizon = {horizon}, K = {K}")

        bl_state = build_initial_state(baseline_wm.dynamics, batch_size, device)
        dr_state = build_initial_state(dualrate_wm.dynamics, batch_size, device)
        action = torch.zeros(batch_size, baseline_wm.dynamics._num_actions, device=device)

        bl_sps, bl_lat = benchmark_rollout_fn(
            lambda s, a, h: rollout_standard(baseline_wm.dynamics, s, a, h),
            bl_state, action, horizon, label="Baseline (standard)"
        )
        dr_std_sps, dr_std_lat = benchmark_rollout_fn(
            lambda s, a, h: rollout_standard(dualrate_wm.dynamics, s, a, h),
            dr_state, action, horizon, label="DualRate (standard)"
        )
        dr_fused_sps, dr_fused_lat = benchmark_rollout_fn(
            lambda s, a, h: rollout_fused(dualrate_wm.dynamics, s, a, h, K),
            dr_state, action, horizon, label="DualRate (fused loop)"
        )

        print(f"  Speedup (standard): {dr_std_sps/bl_sps:.3f}x")
        print(f"  Speedup (fused):    {dr_fused_sps/bl_sps:.3f}x")

        all_results[batch_size] = {
            'baseline_sps': round(bl_sps, 0),
            'dualrate_standard_sps': round(dr_std_sps, 0),
            'dualrate_fused_sps': round(dr_fused_sps, 0),
            'speedup_standard': round(dr_std_sps/bl_sps, 3),
            'speedup_fused': round(dr_fused_sps/bl_sps, 3),
        }

    print(f"\n{'='*50}")
    print("Summary:")
    for bs, r in all_results.items():
        print(f"  batch={bs}: standard={r['speedup_standard']:.3f}x, fused={r['speedup_fused']:.3f}x")

    out_path = pathlib.Path("rollout_chunked_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
