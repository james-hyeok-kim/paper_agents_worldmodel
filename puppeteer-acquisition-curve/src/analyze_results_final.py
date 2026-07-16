"""
R8 최종 분석: AUC_100k 재계산(n=11 seed 확장 + 10M n=5 + gaps/walls 일반화) + 통계 재검정.

방법론(experiment_summary.md와 동일하게 유지):
  - 각 seed의 eval.csv를 0/20k/40k/60k/80k/100k 6개 지점으로 선형보간(np.interp)
  - 그 보간된 곡선의 trapezoid AUC를 100,000으로 나눈 값 = 그 seed의 AUC_100k
  - 조건별 평균/표준편차는 "seed별 AUC_100k 값들"의 평균/표준편차
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOGS = Path("/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer/logs")
OUTDIR = Path("/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results")

GRID = [0, 20000, 40000, 60000, 80000, 100000]


def load_eval_csv(task: str, seed: int, exp_name: str):
    csv_path = LOGS / task / str(seed) / exp_name / "eval.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = ["step", "episode_reward"]
    df = df.sort_values("step").reset_index(drop=True)
    return df


def seed_auc_100k(df: pd.DataFrame) -> float:
    """0~100k 사이 6개 지점(0/20k/.../100k)으로 보간 후 trapezoid AUC / 100000."""
    interp = np.interp(GRID, df["step"], df["episode_reward"])
    return float(np.trapezoid(interp, GRID) / 100000)


def condition_aucs(task: str, exp_name_fmt: str, seeds):
    """exp_name_fmt: '{seed}'를 포함하는 exp_name 템플릿, 예: 'ablation_500000' 또는 'condA_s{seed}'"""
    aucs = {}
    for seed in seeds:
        exp_name = exp_name_fmt.format(seed=seed)
        df = load_eval_csv(task, seed, exp_name)
        if df is None:
            print(f"  WARNING: missing {task}/{seed}/{exp_name}")
            continue
        # 100k까지 데이터가 없는 경우(마지막 step이 100k 미만) 경고
        if df["step"].max() < 100000:
            print(f"  WARNING: {task}/{seed}/{exp_name} max step={df['step'].max():.0f} < 100000")
        aucs[seed] = seed_auc_100k(df)
    return aucs


def summarize(aucs: dict):
    vals = np.array(list(aucs.values()))
    return {
        "seeds": aucs,
        "n": len(vals),
        "mean": float(vals.mean()) if len(vals) else None,
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
    }


def welch_t(a: dict, b: dict, label_a: str, label_b: str):
    va = np.array(list(a["seeds"].values()))
    vb = np.array(list(b["seeds"].values()))
    t, p = stats.ttest_ind(va, vb, equal_var=False)
    d = (va.mean() - vb.mean()) / np.sqrt((va.std(ddof=1) ** 2 + vb.std(ddof=1) ** 2) / 2)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "mean_diff": float(va.mean() - vb.mean()),
        "t_stat": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
        "n_a": len(va),
        "n_b": len(vb),
    }


def welch_anova(groups: list):
    """Welch(1951) 이분산 보정 one-way ANOVA. scipy에 내장 함수가 없어 직접 구현.
    1M 조건의 표준편차(2.362)가 500k(0.814)/3M(0.925)보다 훨씬 커서
    등분산 가정의 표준 f_oneway보다 이쪽이 더 정직한 omnibus test."""
    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    weights = ns / variances
    sum_w = weights.sum()
    grand_mean = (weights * means).sum() / sum_w

    numerator = (weights * (means - grand_mean) ** 2).sum() / (k - 1)
    term = ((1 - weights / sum_w) ** 2 / (ns - 1)).sum()
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * term
    f_stat = numerator / denominator

    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * term)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return {"f_stat": float(f_stat), "df1": float(df1), "df2": float(df2), "p_value": p_value}


def power_analysis(a: dict, b: dict, n_sims=20000, alpha=0.05, seed=0):
    """a,b의 관측된 mean/std를 모수로 가정하고, n=len(a)에서의 post-hoc power를 시뮬레이션."""
    rng = np.random.default_rng(seed)
    va = np.array(list(a["seeds"].values()))
    vb = np.array(list(b["seeds"].values()))
    na, nb = len(va), len(vb)
    ma, mb = va.mean(), vb.mean()
    sa, sb = va.std(ddof=1), vb.std(ddof=1)
    sig = 0
    for _ in range(n_sims):
        sim_a = rng.normal(ma, sa, na)
        sim_b = rng.normal(mb, sb, nb)
        _, p = stats.ttest_ind(sim_a, sim_b, equal_var=False)
        if p < alpha:
            sig += 1
    return sig / n_sims


def main():
    results = {}

    print("=" * 70)
    print("1. corridor: 0-step / 500k / 1M / 3M (n=11) / 10M (n=5)")
    print("=" * 70)

    results["corridor_0step"] = summarize(condition_aucs("corridor", "condB_s{seed}", [1, 2]))
    results["corridor_500k"] = summarize(condition_aucs("corridor", "ablation_500000", range(1, 12)))
    results["corridor_1M"] = summarize(condition_aucs("corridor", "ablation_1000000", range(1, 12)))
    results["corridor_3M"] = summarize(condition_aucs("corridor", "ablation_3000000", range(1, 12)))

    # 10M: seed1,2 = condA_s{seed} (exp1 trained tracker), seed3,4,5 = ablation_10000000
    aucs_10m = {}
    aucs_10m.update(condition_aucs("corridor", "condA_s{seed}", [1, 2]))
    aucs_10m.update(condition_aucs("corridor", "ablation_10000000", [3, 4, 5]))
    results["corridor_10M"] = summarize(aucs_10m)

    for k in ["corridor_0step", "corridor_500k", "corridor_1M", "corridor_3M", "corridor_10M"]:
        r = results[k]
        print(f"{k}: n={r['n']} mean={r['mean']:.3f} std={r['std']:.3f} seeds={r['seeds']}")

    print()
    print("=" * 70)
    print("2. gaps-corridor / walls-corridor 일반화 검증 (n=3 각각)")
    print("=" * 70)
    for task in ["gaps-corridor", "walls-corridor"]:
        prefix = "gaps" if task == "gaps-corridor" else "walls"
        results[f"{prefix}_0step"] = summarize(condition_aucs(task, f"{prefix}_0step", [1, 2, 3]))
        results[f"{prefix}_500k"] = summarize(condition_aucs(task, f"{prefix}_500k", [1, 2, 3]))
        for cond in ["0step", "500k"]:
            r = results[f"{prefix}_{cond}"]
            print(f"{prefix}_{cond}: n={r['n']} mean={r['mean']:.3f} std={r['std']:.3f} seeds={r['seeds']}")

    print()
    print("=" * 70)
    print("3. 통계 재검정")
    print("=" * 70)

    stat_tests = {}

    # 500k/1M/3M 3-way ANOVA (n=11 each)
    v500k = np.array(list(results["corridor_500k"]["seeds"].values()))
    v1m = np.array(list(results["corridor_1M"]["seeds"].values()))
    v3m = np.array(list(results["corridor_3M"]["seeds"].values()))
    f_stat, p_anova = stats.f_oneway(v500k, v1m, v3m)
    stat_tests["anova_500k_1M_3M_n11"] = {"f_stat": float(f_stat), "p_value": float(p_anova)}
    print(f"ANOVA (등분산 가정, 500k/1M/3M, n=11 each): F={f_stat:.3f} p={p_anova:.4f}")

    # 1M의 표준편차(2.362)가 500k(0.814)/3M(0.925)보다 훨씬 커서 등분산 가정이 깨짐
    # -> Welch's ANOVA(이분산 보정)를 정식 omnibus test로 병기
    welch_anova_res = welch_anova([v500k, v1m, v3m])
    stat_tests["welch_anova_500k_1M_3M_n11"] = welch_anova_res
    print(f"Welch's ANOVA (이분산 보정, 500k/1M/3M, n=11 each): "
          f"F={welch_anova_res['f_stat']:.3f} df=({welch_anova_res['df1']:.1f},{welch_anova_res['df2']:.1f}) "
          f"p={welch_anova_res['p_value']:.4f}")

    # pairwise Welch t-tests among 500k/1M/3M (Bonferroni: alpha=0.05/3=0.0167)
    BONFERRONI_ALPHA = 0.05 / 3
    stat_tests["bonferroni_alpha_3_comparisons"] = BONFERRONI_ALPHA
    pairs = [
        ("500k", results["corridor_500k"], "1M", results["corridor_1M"]),
        ("1M", results["corridor_1M"], "3M", results["corridor_3M"]),
        ("500k", results["corridor_500k"], "3M", results["corridor_3M"]),
    ]
    for la, a, lb, b in pairs:
        res = welch_t(a, b, la, lb)
        res["significant_bonferroni"] = bool(res["p_value"] < BONFERRONI_ALPHA)
        stat_tests[f"welch_{la}_vs_{lb}_n11"] = res
        sig_mark = " *(Bonferroni 생존)*" if res["significant_bonferroni"] else ""
        print(f"  {la} vs {lb}: mean_diff={res['mean_diff']:+.3f} t={res['t_stat']:.3f} "
              f"p={res['p_value']:.4f} d={res['cohens_d']:.3f}{sig_mark}")

    # power at n=11 for 500k vs 3M (using observed n=11 mean/std as population params)
    power_n11 = power_analysis(results["corridor_500k"], results["corridor_3M"])
    stat_tests["power_500k_vs_3M_n11"] = power_n11
    print(f"  500k vs 3M 검정력(n=11, 시뮬레이션): {power_n11:.1%}")

    # 3M (n=11) vs 10M (n=5) Welch t-test
    res_3m_10m = welch_t(results["corridor_3M"], results["corridor_10M"], "3M", "10M")
    stat_tests["welch_3M_vs_10M"] = res_3m_10m
    print(f"\n3M(n=11) vs 10M(n=5): mean_diff={res_3m_10m['mean_diff']:+.3f} "
          f"t={res_3m_10m['t_stat']:.3f} p={res_3m_10m['p_value']:.4f} d={res_3m_10m['cohens_d']:.3f}")

    # 500k (n=11) vs 10M (n=5) Welch t-test -- 10M이 500k 수준까지 "회복"하는지 확인
    res_500k_10m = welch_t(results["corridor_500k"], results["corridor_10M"], "500k", "10M")
    stat_tests["welch_500k_vs_10M"] = res_500k_10m
    print(f"500k(n=11) vs 10M(n=5): mean_diff={res_500k_10m['mean_diff']:+.3f} "
          f"t={res_500k_10m['t_stat']:.3f} p={res_500k_10m['p_value']:.4f} d={res_500k_10m['cohens_d']:.3f}")

    # gaps/walls 0-step vs 500k
    for prefix in ["gaps", "walls"]:
        res = welch_t(results[f"{prefix}_500k"], results[f"{prefix}_0step"], f"{prefix}_500k", f"{prefix}_0step")
        stat_tests[f"welch_{prefix}_500k_vs_0step"] = res
        print(f"\n{prefix}-corridor 500k vs 0-step (n=3 각각): mean_diff={res['mean_diff']:+.3f} "
              f"t={res['t_stat']:.3f} p={res['p_value']:.4f} d={res['cohens_d']:.3f}")

    output = {
        "conditions": results,
        "stat_tests": stat_tests,
    }
    out_path = OUTDIR / "final_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
