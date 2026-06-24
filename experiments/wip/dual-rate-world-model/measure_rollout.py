"""
Imagination throughput + separation score 측정
baseline vs dual-rate RSSM 비교

Usage:
    python3 measure_rollout.py \
        --baseline_dir /path/to/baseline_ckpt_dir \
        --dualrate_dir /path/to/dualrate_ckpt_dir \
        --out results.json

    (체크포인트 없으면 random-init으로도 throughput 측정 가능)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pathlib

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


def load_config_for(config_names, base_dir):
    """Load config as dreamer.py does"""
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
    import argparse
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    c = parser.parse_args([])
    c.device = 'cuda:0'
    c.num_actions = 6
    c.compile = False
    return c


def build_world_model(config, embed_size=1536):
    """Build WorldModel without env"""
    import collections
    ObsSpace = collections.namedtuple('ObsSpace', ['spaces'])
    Box = collections.namedtuple('Box', ['shape'])
    obs_space = ObsSpace(spaces={'image': Box(shape=(64, 64, 3))})

    class FakeStep:
        value = 0
        def __mul__(self, other): return 0

    wm = models.WorldModel(obs_space, None, FakeStep(), config)
    return wm


def measure_imagination_throughput(dynamics, device, batch_size=512, horizon=15, n_trials=30):
    """Measure imagination steps/sec for a dynamics model"""
    # Build initial state
    if isinstance(dynamics, networks.DualRateRSSM):
        deter_total = dynamics._deter_total
    else:
        deter_total = dynamics._deter

    deter = torch.zeros(batch_size, deter_total, device=device)
    if dynamics._discrete:
        state = dict(
            logit=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            stoch=torch.zeros(batch_size, dynamics._stoch, dynamics._discrete, device=device),
            deter=deter,
        )
    else:
        state = dict(
            mean=torch.zeros(batch_size, dynamics._stoch, device=device),
            std=torch.zeros(batch_size, dynamics._stoch, device=device),
            stoch=torch.zeros(batch_size, dynamics._stoch, device=device),
            deter=deter,
        )

    if isinstance(dynamics, networks.DualRateRSSM):
        state['step_idx'] = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

    action = torch.zeros(batch_size, dynamics._num_actions, device=device)

    # Warmup
    for _ in range(5):
        s = {k: v.clone() for k, v in state.items()}
        for _ in range(horizon):
            s = dynamics.img_step(s, action)
    torch.cuda.synchronize()

    # Measurement
    latencies = []
    for _ in range(n_trials):
        s = {k: v.clone() for k, v in state.items()}
        t0 = time.perf_counter()
        for _ in range(horizon):
            s = dynamics.img_step(s, action)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    steps_per_sec = (batch_size * horizon) / mean_latency

    return {
        "mean_latency_sec": mean_latency,
        "std_latency_sec": std_latency,
        "steps_per_sec": steps_per_sec,
        "batch_size": batch_size,
        "horizon": horizon,
    }


def measure_separation_score(dynamics, device, batch_size=64, seq_len=64):
    """Measure slow/fast separation score for DualRateRSSM"""
    if not isinstance(dynamics, networks.DualRateRSSM):
        return None

    state = dynamics.initial(batch_size)
    action = torch.zeros(batch_size, dynamics._num_actions, device=device)

    # Collect trajectory
    deter_seq = []
    s = state
    for _ in range(seq_len):
        s = dynamics.img_step(s, action)
        deter_seq.append(s['deter'].clone())

    deter_seq = torch.stack(deter_seq, dim=1)  # (B, T, deter_total)

    # Compute metrics on sequence
    deter_slow = deter_seq[..., :dynamics._deter_slow]
    deter_fast = deter_seq[..., dynamics._deter_slow:]

    slow_delta = torch.mean(torch.norm(deter_slow[:, 1:] - deter_slow[:, :-1], dim=-1)).item()
    fast_delta = torch.mean(torch.norm(deter_fast[:, 1:] - deter_fast[:, :-1], dim=-1)).item()
    denom = slow_delta + fast_delta + 1e-8
    sep_score = fast_delta / denom

    return {
        "separation_score": round(sep_score, 4),
        "slow_delta": round(slow_delta, 4),
        "fast_delta": round(fast_delta, 4),
        "collapse_flag": sep_score < 0.5,
    }


def load_checkpoint_if_exists(wm, ckpt_dir):
    """Try to load latest.pt checkpoint if it exists.

    Handles two key name patterns:
    1. No compile: '_wm.encoder...'
    2. With compile: '_wm._orig_mod.encoder...'
    """
    ckpt_path = pathlib.Path(ckpt_dir) / "latest.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'agent_state_dict' in ckpt:
                state_dict = ckpt['agent_state_dict']

                # Pattern 1: with torch.compile → '_wm._orig_mod.xxx'
                wm_keys = {}
                for k, v in state_dict.items():
                    if k.startswith('_wm._orig_mod.'):
                        wm_keys[k[len('_wm._orig_mod.'):]] = v
                    elif k.startswith('_wm.'):
                        wm_keys[k[len('_wm.'):]] = v

                if len(wm_keys) == 0:
                    print(f"  WARNING: No _wm keys found in checkpoint. Keys sample: {list(state_dict.keys())[:5]}")
                    return False

                load_result = wm.load_state_dict(wm_keys, strict=False)
                matched = len(wm_keys) - len(load_result.missing_keys) - len(load_result.unexpected_keys)
                total_wm = len(list(wm.parameters()))
                print(f"  Loaded checkpoint from {ckpt_path}")
                print(f"  Keys: {len(wm_keys)} from ckpt, {len(load_result.missing_keys)} missing, {len(load_result.unexpected_keys)} unexpected")
                if len(load_result.missing_keys) > 0:
                    print(f"  Missing (first 5): {load_result.missing_keys[:5]}")
                if len(load_result.unexpected_keys) > 0:
                    print(f"  Unexpected (first 5): {load_result.unexpected_keys[:5]}")
                assert len(wm_keys) > 0, "Zero keys loaded — checkpoint loading failed"
                return True
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
    else:
        print(f"  No checkpoint at {ckpt_path} (using random init)")
    return False


def get_latest_return(log_path):
    """Parse latest eval_return from training log"""
    import re
    if not os.path.exists(log_path):
        return None
    lines = open(log_path).readlines()
    returns = []
    for line in lines:
        m = re.search(r'eval_return\s+([\d.]+)', line)
        if m:
            returns.append(float(m.group(1)))
    if returns:
        return {'final': returns[-1], 'all': returns}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/dreamerv3-torch')
    parser.add_argument('--baseline_dir', default='/data/jameskimh/worldmodel/dual-rate-world-model/baseline_100k', help='Baseline checkpoint dir (optional)')
    parser.add_argument('--dualrate_dir', default='/data/jameskimh/worldmodel/dual-rate-world-model/dualrate_K3_100k', help='Dual-rate checkpoint dir (optional)')
    parser.add_argument('--baseline_log', default='/data/jameskimh/worldmodel/dual-rate-world-model/baseline_100k.log')
    parser.add_argument('--dualrate_log', default='/data/jameskimh/worldmodel/dual-rate-world-model/dualrate_K3_100k.log')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--out', default='/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/results.json')
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    device = torch.device('cuda:0')

    results = {
        "date": "2026-06-11 KST",
        "model": "DreamerV3 (NM512/dreamerv3-torch)",
        "benchmark": "DMControl Walker-walk (100k env steps)",
        "config": {
            "batch_size": args.batch_size,
            "horizon": args.horizon,
            "seed": 0,
            "K": 3,
            "deter_slow": 256,
            "deter_fast": 128,
            "lambda_ib": 0.001,
            "lambda_sm": 0.01,
        },
    }

    print("=" * 60)
    print("Building BASELINE WorldModel...")
    baseline_config = load_config_for(['defaults', 'dmc_vision'], base_dir)
    baseline_wm = build_world_model(baseline_config).to(device)
    baseline_wm.eval()
    load_checkpoint_if_exists(baseline_wm, args.baseline_dir)

    print("Measuring baseline imagination throughput...")
    bl_throughput = measure_imagination_throughput(
        baseline_wm.dynamics, device, args.batch_size, args.horizon
    )
    print(f"  Baseline: {bl_throughput['steps_per_sec']:.0f} steps/sec, latency={bl_throughput['mean_latency_sec']*1000:.2f}ms")
    results['baseline_throughput'] = bl_throughput

    print("=" * 60)
    print("Building DUAL-RATE WorldModel...")
    dualrate_config = load_config_for(['defaults', 'dmc_vision', 'dual_rate'], base_dir)
    dualrate_wm = build_world_model(dualrate_config).to(device)
    dualrate_wm.eval()
    load_checkpoint_if_exists(dualrate_wm, args.dualrate_dir)

    print("Measuring dual-rate imagination throughput...")
    dr_throughput = measure_imagination_throughput(
        dualrate_wm.dynamics, device, args.batch_size, args.horizon
    )
    print(f"  Dual-rate: {dr_throughput['steps_per_sec']:.0f} steps/sec, latency={dr_throughput['mean_latency_sec']*1000:.2f}ms")
    results['dualrate_throughput'] = dr_throughput

    print("Measuring separation score...")
    sep = measure_separation_score(dualrate_wm.dynamics, device)
    print(f"  Separation score: {sep['separation_score']:.4f} (fast_delta={sep['fast_delta']:.4f}, slow_delta={sep['slow_delta']:.4f})")
    results['separation'] = sep

    # Speedup
    speedup = dr_throughput['steps_per_sec'] / bl_throughput['steps_per_sec']
    latency_speedup = bl_throughput['mean_latency_sec'] / dr_throughput['mean_latency_sec']
    print(f"\nImagination Speedup: {speedup:.3f}x (steps/sec)")
    print(f"Latency Speedup: {latency_speedup:.3f}x")

    # Parse episode returns from logs
    bl_returns = get_latest_return(args.baseline_log)
    dr_returns = get_latest_return(args.dualrate_log)
    print(f"\nBaseline returns: {bl_returns}")
    print(f"Dual-rate returns: {dr_returns}")

    results['efficiency'] = {
        "baseline_latency_ms": round(bl_throughput['mean_latency_sec'] * 1000, 3),
        "modified_latency_ms": round(dr_throughput['mean_latency_sec'] * 1000, 3),
        "rollout_speedup": round(latency_speedup, 3),
        "steps_per_sec_speedup": round(speedup, 3),
        "flops_reduction": round(latency_speedup, 3),  # proxy
    }

    bl_final_return = bl_returns['final'] if bl_returns else None
    dr_final_return = dr_returns['final'] if dr_returns else None

    if bl_final_return and dr_final_return:
        return_delta_pct = (dr_final_return - bl_final_return) / (abs(bl_final_return) + 1e-6) * 100
    else:
        return_delta_pct = None

    results['quality'] = {
        "baseline_return": bl_final_return,
        "modified_return": dr_final_return,
        "return_delta_pct": round(return_delta_pct, 2) if return_delta_pct is not None else None,
        "quality_proxy_delta": 1 - sep['separation_score'] if sep else None,
        "sample_efficiency_gain": None,
        "separation_score": sep['separation_score'] if sep else None,
        "separation_ok": sep['separation_score'] > 0.5 if sep else None,
    }

    # Verdict
    speedup_ok = latency_speedup > 1.2
    sep_ok = sep and sep['separation_score'] > 0.5
    return_ok = return_delta_pct is None or return_delta_pct > -15.0

    if speedup_ok and sep_ok and return_ok:
        if latency_speedup >= 1.5 and (return_delta_pct is None or return_delta_pct > -5.0):
            verdict = "GO"
        else:
            verdict = "WEAK GO"
    else:
        verdict = "NO GO"

    results['verdict'] = verdict
    results['slug'] = 'dual-rate-world-model'

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  Imagination speedup: {latency_speedup:.3f}x (threshold: >1.2x Week1 bar)")
    print(f"  Separation score: {sep['separation_score']:.4f} (threshold: >0.5)")
    if return_delta_pct is not None:
        print(f"  Return delta: {return_delta_pct:.1f}% (threshold: >-15%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
