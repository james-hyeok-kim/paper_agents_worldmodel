"""
AUC_100k / T_50 계산 및 결과 요약 스크립트.
Condition A (trained tracker) vs Condition B (random tracker) 비교.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

PUPPETEER_LOGS = Path("/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer/logs/corridor")
OUTDIR = Path("/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve")

CONDITIONS = {
    "condA": {"seeds": [1, 2], "exp_name": "condA_s{seed}", "label": "Trained tracker (Cond. A)"},
    "condB": {"seeds": [1, 2], "exp_name": "condB_s{seed}", "label": "Random tracker (Cond. B)"},
}


def load_eval_csv(seed: int, exp_name: str) -> pd.DataFrame:
    csv_path = PUPPETEER_LOGS / str(seed) / exp_name / "eval.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    df.columns = ["step", "episode_reward"]
    df = df.sort_values("step").reset_index(drop=True)
    return df


def compute_auc(df: pd.DataFrame, max_step: int = 100_000) -> float:
    """trapezoid AUC up to max_step."""
    sub = df[df["step"] <= max_step].copy()
    if len(sub) < 2:
        return float(sub["episode_reward"].mean()) if len(sub) == 1 else 0.0
    return float(np.trapz(sub["episode_reward"], sub["step"]) / max_step)


def compute_t50(df: pd.DataFrame, target_return: float) -> int:
    """First step where episode_reward >= 0.5 * target_return."""
    threshold = 0.5 * target_return
    hits = df[df["episode_reward"] >= threshold]
    if hits.empty:
        return None
    return int(hits.iloc[0]["step"])


def main():
    results = {}

    for cond_key, cond_cfg in CONDITIONS.items():
        dfs = []
        for seed in cond_cfg["seeds"]:
            exp = cond_cfg["exp_name"].format(seed=seed)
            df = load_eval_csv(seed, exp)
            if df is not None:
                df["seed"] = seed
                dfs.append(df)
                print(f"  Loaded {cond_key} seed={seed}: {len(df)} evals, "
                      f"final return={df['episode_reward'].iloc[-1]:.2f}")

        if not dfs:
            print(f"  No data for {cond_key}")
            results[cond_key] = None
            continue

        # 공통 step 포인트로 interpolate (평균 계산용)
        all_steps = sorted(set(s for df in dfs for s in df["step"].tolist()))
        interpolated = []
        for df in dfs:
            interp = np.interp(all_steps, df["step"], df["episode_reward"])
            interpolated.append(interp)
        mean_returns = np.mean(interpolated, axis=0)
        std_returns = np.std(interpolated, axis=0) if len(interpolated) > 1 else np.zeros_like(mean_returns)

        mean_df = pd.DataFrame({"step": all_steps, "episode_reward": mean_returns})
        auc_100k = compute_auc(mean_df, 100_000)
        final_return = float(mean_returns[-1])

        results[cond_key] = {
            "label": cond_cfg["label"],
            "n_seeds": len(dfs),
            "auc_100k": auc_100k,
            "final_return_200k": final_return,
            "final_std_200k": float(std_returns[-1]) if len(interpolated) > 1 else 0.0,
            "steps": all_steps,
            "mean_returns": mean_returns.tolist(),
            "std_returns": std_returns.tolist(),
        }

    # Gate 판정
    if results.get("condA") and results.get("condB"):
        a = results["condA"]
        b = results["condB"]
        ratio = a["auc_100k"] / max(abs(b["auc_100k"]), 1e-6)
        verdict = "PASS" if ratio >= 2.0 else "FAIL"

        # T_50 계산 (Condition A final 기준)
        a_final = a["final_return_200k"]
        a_df = pd.DataFrame({"step": a["steps"], "episode_reward": a["mean_returns"]})
        b_df = pd.DataFrame({"step": b["steps"], "episode_reward": b["mean_returns"]})
        t50_a = compute_t50(a_df, a_final)
        t50_b = compute_t50(b_df, a_final)

        print("\n" + "="*60)
        print("ACQUISITION CURVE ANALYSIS")
        print("="*60)
        print(f"Condition A  AUC_100k: {a['auc_100k']:.3f}  Final(200k): {a['final_return_200k']:.2f} ± {a['final_std_200k']:.2f}")
        print(f"Condition B  AUC_100k: {b['auc_100k']:.3f}  Final(200k): {b['final_return_200k']:.2f} ± {b['final_std_200k']:.2f}")
        print(f"AUC ratio (A/B):       {ratio:.2f}×")
        print(f"T_50 (A): {t50_a}  T_50 (B): {t50_b}")
        print(f"\nVERDICT: {verdict}")
        if verdict == "PASS":
            print("  → Trained tracker provides significant acquisition advantage")
        else:
            print("  → Tracker quality does not significantly affect acquisition speed")

        # JSON 저장
        output = {
            "date": "2026-06-23 KST",
            "condA": {k: v for k, v in a.items() if k not in ("steps", "mean_returns", "std_returns")},
            "condB": {k: v for k, v in b.items() if k not in ("steps", "mean_returns", "std_returns")},
            "comparison": {
                "auc_ratio_A_over_B": round(ratio, 3),
                "t50_condA_steps": t50_a,
                "t50_condB_steps": t50_b,
                "verdict": verdict,
                "gate_threshold": "A AUC_100k >= 2x B AUC_100k",
            }
        }
        out_path = OUTDIR / "analysis_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")
    else:
        print("WARNING: Could not compare conditions — missing data")


if __name__ == "__main__":
    main()
