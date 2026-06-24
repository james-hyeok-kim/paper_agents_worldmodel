"""
Dual-Rate World Model — Paper Figures
Generates 4 plots:
  1. eval_return: baseline vs dualrate (multi-seed, shaded std)
  2. FPS comparison bar chart
  3. separation_score + fast/slow delta trajectory
  4. K sweep Pareto: FLOPs reduction vs late-stage quality

Usage:
  python visualize_results.py [--logbase /data/jameskimh/worldmodel/dual-rate-paper] [--output figures/]
"""
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ─── Data loading ──────────────────────────────────────────────────────────────

def load_metrics(logdir):
    p = Path(logdir) / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def get_eval_series(data):
    """Returns list of (step, eval_return) sorted by step."""
    return sorted([(d["step"], d["eval_return"]) for d in data if "eval_return" in d])


def get_fps_series(data):
    return sorted([(d["step"], d["fps"]) for d in data if "fps" in d and d["fps"] > 0])


def get_separation_series(data):
    keys = ["dual_rate/separation_score", "dual_rate/fast_delta", "dual_rate/slow_delta"]
    return sorted([
        (d["step"], d["dual_rate/separation_score"], d["dual_rate/fast_delta"], d["dual_rate/slow_delta"])
        for d in data if all(k in d for k in keys)
    ])


def flops_reduction(K, deter_slow=256, deter_fast=128, deter_baseline=512):
    """Analytical FLOPs reduction for dual-rate vs baseline."""
    # GRU FLOPs ≈ 3 * (inp_dim + hidden_dim) * hidden_dim
    # Baseline: inp_dim=38, hidden=512 → 3*(38+512)*512 = 845760
    inp_base = 38
    flops_base = 3 * (inp_base + deter_baseline) * deter_baseline
    # DualRate fast: inp=38+deter_slow, hidden=deter_fast (every step)
    flops_fast = 3 * (inp_base + deter_slow + deter_fast) * deter_fast
    # DualRate slow: inp=deter_fast, hidden=deter_slow (every K steps)
    flops_slow_per_K = 3 * (deter_fast + deter_slow) * deter_slow
    flops_dual_avg = flops_fast + flops_slow_per_K / K
    return flops_base / flops_dual_avg


# ─── Late-stage mean helper ─────────────────────────────────────────────────────

def late_stage_mean(eval_series, min_step=75000):
    vals = [v for s, v in eval_series if s >= min_step]
    return np.mean(vals) if vals else None


# ─── Plot 1: eval_return multi-seed ────────────────────────────────────────────

def plot_eval_return_seeds(logbase, output_dir, existing_seed0_base, existing_seed0_dr):
    """Plot eval_return for baseline vs dualrate across seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Collect per-seed data
    baseline_seeds = {}
    dualrate_seeds = {}

    # Existing seed=0
    d = load_metrics(existing_seed0_base)
    if d: baseline_seeds[0] = get_eval_series(d)
    d = load_metrics(existing_seed0_dr)
    if d: dualrate_seeds[0] = get_eval_series(d)

    # New seeds
    for seed in [1, 2]:
        d = load_metrics(Path(logbase) / f"baseline_seed{seed}")
        if d: baseline_seeds[seed] = get_eval_series(d)
        d = load_metrics(Path(logbase) / f"dualrate_K3_seed{seed}")
        if d: dualrate_seeds[seed] = get_eval_series(d)

    def plot_band(seeds_dict, color, label):
        if not seeds_dict: return
        # Interpolate all seeds to common step grid
        valid = {k: v for k, v in seeds_dict.items() if v}
        if not valid: return
        all_steps = sorted(set(s for series in valid.values() for s, v in series))
        if len(all_steps) < 2: return
        seeds_dict = valid
        interp_vals = []
        for seed_id, series in seeds_dict.items():
            steps, vals = zip(*series)
            interp = np.interp(all_steps, steps, vals)
            interp_vals.append(interp)
        arr = np.array(interp_vals)
        mean = arr.mean(0)
        std = arr.std(0)
        ax.plot(all_steps, mean, color=color, label=f"{label} (n={len(seeds_dict)})", lw=2)
        ax.fill_between(all_steps, mean - std, mean + std, alpha=0.2, color=color)

    plot_band(baseline_seeds, "#2196F3", "Baseline (D=512)")
    plot_band(dualrate_seeds, "#FF5722", "Dual-Rate K=3 (D=384)")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("DMControl Walker-Walk: Eval Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "fig1_eval_return.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close()


# ─── Plot 2: Separation score trajectory ───────────────────────────────────────

def plot_separation(existing_seed0_dr, output_dir):
    """Plot separation_score and fast/slow delta over training."""
    d = load_metrics(existing_seed0_dr)
    sep_data = get_separation_series(d)
    if not sep_data:
        print("No separation data found")
        return

    steps = [x[0] for x in sep_data]
    sep  = [x[1] for x in sep_data]
    fd   = [x[2] for x in sep_data]
    sd   = [x[3] for x in sep_data]
    ratio = [f/max(s, 1e-6) for f, s in zip(fd, sd)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(steps, sep, "g-o", lw=2, markersize=4, label="separation_score")
    ax1.axhline(0.5, ls="--", color="gray", alpha=0.6, label="threshold=0.5")
    ax1.set_xlabel("Step"); ax1.set_ylabel("separation_score")
    ax1.set_title("Fast/Slow Separation Score"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(steps, fd, "r-o", lw=2, markersize=4, label="fast_delta")
    ax2.plot(steps, sd, "b-o", lw=2, markersize=4, label="slow_delta")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Δ latent per step")
    ax2.set_title("Fast vs Slow Branch Dynamics"); ax2.legend(); ax2.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(steps, ratio, "k--", lw=1.5, alpha=0.5, label="ratio")
    ax2b.set_ylabel("fast/slow ratio", color="gray")
    ax2b.tick_params(axis="y", labelcolor="gray")

    plt.tight_layout()
    out = Path(output_dir) / "fig2_separation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close()


# ─── Plot 3: K sweep Pareto ─────────────────────────────────────────────────────

def plot_k_sweep_pareto(logbase, existing_seed0_dr, existing_seed0_base, output_dir):
    """Plot FLOPs reduction vs late-stage quality for K=2,3,4,6 + baseline."""
    logbase = Path(logbase)

    runs = {
        "K=2": (logbase / "dualrate_K2_seed0", 2),
        "K=3\n(proposed)": (existing_seed0_dr, 3),
        "K=4": (logbase / "dualrate_K4_seed0", 4),
        "K=6": (logbase / "dualrate_K6_seed0", 6),
    }
    baseline_d = load_metrics(existing_seed0_base)
    baseline_late = late_stage_mean(get_eval_series(baseline_d)) if baseline_d else None

    fig, ax = plt.subplots(figsize=(6, 4.5))

    if baseline_late:
        ax.axhline(baseline_late, ls="--", color="#2196F3", lw=1.5, label=f"Baseline ({baseline_late:.0f})")

    for label, (logdir, K) in runs.items():
        d = load_metrics(logdir)
        if not d:
            print(f"  {label}: no data yet")
            continue
        late = late_stage_mean(get_eval_series(d))
        if late is None: continue
        flops_r = flops_reduction(K)
        size_pt = 120 if "proposed" in label else 80
        color = "#FF5722" if "proposed" in label else "#FF8A65"
        ax.scatter(flops_r, late, s=size_pt, color=color, zorder=5)
        ax.annotate(label, (flops_r, late), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("FLOPs Reduction (×)")
    ax.set_ylabel("Late-Stage Eval Return (75k-105k mean)")
    ax.set_title("K Sweep: FLOPs vs Quality Pareto")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "fig3_k_sweep.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close()


# ─── Plot 4: Multi-environment bar chart ────────────────────────────────────────

def plot_multi_env(logbase, existing_seed0_base, existing_seed0_dr, output_dir):
    """Bar chart: baseline vs dualrate across environments."""
    logbase = Path(logbase)

    envs = {
        "Walker-Walk": (existing_seed0_base, existing_seed0_dr),
        "Cheetah-Run": (logbase / "baseline_cheetah_seed0", logbase / "dualrate_cheetah_seed0"),
        "Hopper-Hop":  (logbase / "baseline_hopper_seed0",  logbase / "dualrate_hopper_seed0"),
    }

    env_names = []
    baseline_lates = []
    dualrate_lates = []

    for env_name, (b_dir, d_dir) in envs.items():
        b_d = load_metrics(b_dir)
        d_d = load_metrics(d_dir)
        b_late = late_stage_mean(get_eval_series(b_d)) if b_d else None
        d_late = late_stage_mean(get_eval_series(d_d)) if d_d else None
        if b_late and d_late:
            env_names.append(env_name)
            baseline_lates.append(b_late)
            dualrate_lates.append(d_late)

    if not env_names:
        print("No multi-env data yet")
        return

    x = np.arange(len(env_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width/2, baseline_lates, width, label="Baseline (D=512)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, dualrate_lates, width, label="Dual-Rate K=3 (D=384)", color="#FF5722", alpha=0.85)

    # Relative % labels
    for i, (b, d) in enumerate(zip(baseline_lates, dualrate_lates)):
        diff = (d - b) / max(b, 1) * 100
        color = "green" if diff >= 0 else "red"
        ax.annotate(f"{diff:+.0f}%", xy=(x[i] + width/2, d), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9)
    ax.set_ylabel("Late-Stage Eval Return (75k-105k mean)")
    ax.set_title("Multi-Environment Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "fig4_multi_env.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logbase", default="/data/jameskimh/worldmodel/dual-rate-paper")
    parser.add_argument("--seed0_base", default="/data/jameskimh/worldmodel/dual-rate-world-model/baseline_100k")
    parser.add_argument("--seed0_dr", default="/data/jameskimh/worldmodel/dual-rate-world-model/dualrate_K3_100k")
    parser.add_argument("--output", default="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/figures")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures → {args.output}")

    print("\n[1] eval_return multi-seed...")
    plot_eval_return_seeds(args.logbase, args.output, args.seed0_base, args.seed0_dr)

    print("\n[2] Separation score trajectory...")
    plot_separation(args.seed0_dr, args.output)

    print("\n[3] K sweep Pareto...")
    plot_k_sweep_pareto(args.logbase, args.seed0_dr, args.seed0_base, args.output)

    print("\n[4] Multi-environment bar chart...")
    plot_multi_env(args.logbase, args.seed0_base, args.seed0_dr, args.output)

    print("\nDone. Check figures/")


if __name__ == "__main__":
    main()
