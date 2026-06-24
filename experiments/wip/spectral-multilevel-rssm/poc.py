"""
PoC: Spectral Multi-Level RSSM
Kill criterion: RSSM latent autocorrelation spectrum has ≥3 frequency modes
If BIMODAL → deprioritize (reduces to dual-rate L=2, no novelty)
If ≥3 modes → proceed to speedup test

Step 1: Train RSSM on 3-timescale synthetic data
Step 2: Measure per-dimension temporal autocorrelation
Step 3: Count spectral modes (≥3 = PROCEED, bimodal = KILL)
Step 4 (if proceed): Test if L=3 band skip beats best L=2 (dual-rate) Pareto
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import mode as scipy_mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D = 64       # latent dim (smaller for fast training)
Z = 16       # observation dim (matches 3-scale data)
H = 30       # trajectory length for autocorrelation
N_TRAIN = 3000
N_EPOCHS = 40
LR = 1e-3
B = 128

# Band skip periods to test
BAND_CONFIGS_L2 = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4)]  # L=2 (dual-rate like)
BAND_CONFIGS_L3 = [(1, 2, 4), (1, 2, 6), (1, 3, 6), (1, 2, 5), (1, 3, 9)]  # L=3 (spectral)

# ─── 1. Generate 3-scale synthetic data ──────────────────────────────────────

def generate_3scale_data(n_trajs, H_len=H, d_obs=Z):
    """
    3 time scales:
    - Background (very slow, K_bg=10): changes every 10 steps
    - Drift (medium, K_mid=4): changes every 4 steps
    - Fluctuation (fast, K_fast=1): changes every step
    Each contributes to d_obs/3 dimensions
    """
    d_bg = d_obs // 3
    d_mid = d_obs // 3
    d_fast = d_obs - d_bg - d_mid

    trajs = []
    for _ in range(n_trajs):
        bg_state = torch.randn(d_bg) * 0.5
        mid_state = torch.randn(d_mid) * 0.5

        obs_seq = []
        for t in range(H_len):
            if t % 10 == 0:
                bg_state = bg_state * 0.7 + torch.randn(d_bg) * 0.3
            if t % 4 == 0:
                mid_state = mid_state * 0.5 + torch.randn(d_mid) * 0.5
            fast_state = torch.randn(d_fast)
            obs = torch.cat([bg_state, mid_state, fast_state])
            obs_seq.append(obs)
        trajs.append(torch.stack(obs_seq))  # (H, d_obs)

    return torch.stack(trajs)  # (N, H, d_obs)


# ─── 2. Simple RSSM for training ─────────────────────────────────────────────

class SimpleRSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(Z, D)
        self.gru = nn.GRUCell(D, D)
        self.decoder = nn.Linear(D, Z)

    def rollout(self, obs_seq):
        """obs_seq: (B, T, Z) → returns h_seq: (B, T, D)"""
        B, T, _ = obs_seq.shape
        h = torch.zeros(B, D, device=obs_seq.device)
        hs = []
        for t in range(T):
            x = torch.tanh(self.encoder(obs_seq[:, t]))
            h = self.gru(x, h)
            hs.append(h)
        return torch.stack(hs, dim=1)  # (B, T, D)

    def decode(self, h):
        return self.decoder(h)


def train_rssm(data):
    rssm = SimpleRSSM().to(DEVICE)
    opt = torch.optim.Adam(rssm.parameters(), lr=LR)
    data = data.to(DEVICE)

    print(f"  Training RSSM on {len(data)} trajectories, {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(data))
        total_loss = 0; nb = 0
        for i in range(0, len(data), B):
            idx = perm[i:i+B]
            batch = data[idx]  # (B, H, Z)
            h_seq = rssm.rollout(batch)  # (B, H, D)
            # Predict next observation
            pred = rssm.decode(h_seq[:, :-1])  # (B, H-1, Z)
            target = batch[:, 1:]               # (B, H-1, Z)
            loss = (pred - target).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: loss={total_loss/nb:.5f}")

    return rssm


# ─── 3. Measure per-dimension temporal autocorrelation ───────────────────────

def compute_autocorrelation(h_seq, max_lag=15):
    """
    h_seq: (N_total, D) — concatenated all steps
    Actually we want (N_trajs, T, D) → correlate over T dimension
    Returns: (D, max_lag) autocorrelation per dimension per lag
    """
    # h_seq: (N_trajs, T, D)
    N, T, D = h_seq.shape

    # Normalize per trajectory per dimension
    h_norm = h_seq - h_seq.mean(dim=1, keepdim=True)
    h_std = h_norm.std(dim=1, keepdim=True).clamp(min=1e-6)
    h_norm = h_norm / h_std

    acf = torch.zeros(D, max_lag + 1)
    acf[:, 0] = 1.0

    for lag in range(1, max_lag + 1):
        if lag < T:
            x1 = h_norm[:, :-lag, :]   # (N, T-lag, D)
            x2 = h_norm[:, lag:, :]     # (N, T-lag, D)
            corr = (x1 * x2).mean(dim=(0, 1))  # (D,)
            acf[:, lag] = corr

    return acf  # (D, max_lag+1)


def estimate_characteristic_timescale(acf, threshold=0.5):
    """For each dim, find the lag where ACF first drops below threshold."""
    D, L = acf.shape
    timescales = []
    for d in range(D):
        ac = acf[d] if isinstance(acf[d], np.ndarray) else np.array(acf[d])
        # Find first lag where ac < threshold
        lags_below = np.where(ac < threshold)[0]
        if len(lags_below) > 0:
            timescales.append(float(lags_below[0]))
        else:
            timescales.append(float(L - 1))
    return np.array(timescales)


def count_spectral_modes(timescales, bandwidth=0.5):
    """
    Count number of distinct modes in the timescale distribution.
    Uses histogram + peak detection.
    """
    counts, bin_edges = np.histogram(timescales, bins=15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(counts.astype(float), sigma=1.0)

    # Find peaks
    peaks, props = find_peaks(smooth, height=smooth.max() * 0.1, distance=2)

    return len(peaks), timescales, bin_centers, smooth, peaks


# ─── 4. Band-skip speedup test ────────────────────────────────────────────────

def simulate_band_skip(rssm, data, band_config, n_samples=500, n_reps=50):
    """
    Simulate band-skip rollout.
    band_config: list of (dims_in_band, period_K) pairs

    Speedup is measured by counting skipped GRU calls.
    Quality is measured by rollout MSE vs full sequential.
    """
    # For speedup: theoretical (counting calls)
    total_dims = D
    # Assign dims to bands by timescale
    timescales = band_config["timescales"]  # pre-computed
    periods = band_config["periods"]  # K per band
    band_assignments = band_config["assignments"]  # (D,) → band index

    # Theoretical speedup: fraction of GRU calls saved
    avg_call_fraction = 0.0
    for b_idx, K_b in enumerate(periods):
        n_dims_b = (band_assignments == b_idx).sum()
        avg_call_fraction += (n_dims_b / total_dims) * (1.0 / K_b)

    theoretical_speedup = 1.0 / avg_call_fraction

    # Quality: simulate rollout with band-skip carry-forward
    rssm.eval()
    data_subset = data[:n_samples].to(DEVICE)

    with torch.no_grad():
        # Full sequential (ground truth)
        h_full = rssm.rollout(data_subset)  # (n, H, D)

        # Band-skip rollout
        B_n, T, _ = data_subset.shape
        h = torch.zeros(B_n, D, device=DEVICE)
        h_skip_seq = []

        for t in range(T):
            x = torch.tanh(rssm.encoder(data_subset[:, t]))

            # Sub-GRU: combine fast+slow bands
            new_h = h.clone()
            for b_idx, K_b in enumerate(periods):
                if t % K_b == 0:
                    # This band gets updated
                    band_mask = (band_assignments == b_idx)
                    # Full GRU on full input, but only update band dimensions
                    h_new_full = rssm.gru(x, h)
                    new_h[:, band_mask] = h_new_full[:, band_mask]
                # else: carry forward (no update needed)
            h = new_h
            h_skip_seq.append(h)

        h_skip = torch.stack(h_skip_seq, dim=1)  # (n, H, D)

        # Quality: normalized MSE
        ref = h_full.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mse = ((h_skip - h_full) / ref).pow(2).mean().item()

    return theoretical_speedup, mse


def test_all_band_configs(rssm, data, timescales, band_configs, L_label):
    """Test all band configs and find best Pareto point."""
    print(f"\n  Testing L={L_label} band configs...")

    results = []
    for config in band_configs:
        # Assign dims to bands by timescale quantiles
        n_bands = len(config)
        quantile_edges = [0] + [np.percentile(timescales, 100*i/n_bands) for i in range(1, n_bands)] + [float('inf')]

        assignments = torch.zeros(D, dtype=torch.long)
        for b_idx in range(n_bands):
            lo, hi = quantile_edges[b_idx], quantile_edges[b_idx + 1]
            mask = (timescales >= lo) & (timescales < hi)
            assignments[mask] = b_idx

        band_config = {
            "periods": config,
            "assignments": assignments,
            "timescales": timescales,
        }

        speedup, mse = simulate_band_skip(rssm, data, band_config)
        results.append({
            "periods": config,
            "theoretical_speedup": speedup,
            "quality_mse": mse,
            "quality_pass": mse < 0.05,
            "speed_pass": speedup > 1.5,
        })
        print(f"    K={config}: speedup={speedup:.3f}x, mse={mse:.5f} ({'PASS' if mse<0.05 else 'FAIL'} quality)")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Spectral Multi-Level RSSM PoC ===\n")

    # 1. Generate 3-timescale synthetic data
    print("[1] Generating 3-scale synthetic data...")
    data = generate_3scale_data(N_TRAIN, H_len=H, d_obs=Z)
    print(f"  Data shape: {data.shape}")

    # 2. Train RSSM
    print("\n[2] Training RSSM on 3-scale data...")
    t0 = time.time()
    rssm = train_rssm(data)
    rssm.eval()
    print(f"  Training done in {time.time()-t0:.1f}s")

    # 3. Extract latent trajectories and measure autocorrelation
    print("\n[3] Measuring per-dimension temporal autocorrelation...")
    with torch.no_grad():
        test_data = data[:500].to(DEVICE)
        h_seq = rssm.rollout(test_data).cpu()  # (500, H, D)

    acf = compute_autocorrelation(h_seq, max_lag=15)  # (D, 16)
    print(f"  ACF shape: {acf.shape}")
    print(f"  ACF at lag=0: mean={acf[:,0].mean():.3f}")
    print(f"  ACF at lag=1: mean={acf[:,1].mean():.3f}")
    print(f"  ACF at lag=5: mean={acf[:,5].mean():.3f}")

    # 4. Estimate characteristic timescales
    timescales_arr = estimate_characteristic_timescale(acf.numpy(), threshold=0.5)
    print(f"\n  Timescale distribution:")
    print(f"    min={timescales_arr.min():.2f}, max={timescales_arr.max():.2f}")
    print(f"    mean={timescales_arr.mean():.2f}, std={timescales_arr.std():.2f}")
    print(f"    quantiles: {np.percentile(timescales_arr, [25,50,75,90])}")

    # 5. Count spectral modes — the KILL criterion
    print("\n[4] Counting spectral modes (kill criterion: ≥3 modes = PROCEED)...")
    n_modes, ts, bins, smooth_hist, peaks = count_spectral_modes(timescales_arr)

    print(f"  Number of distinct modes: {n_modes}")
    print(f"  Mode locations (characteristic timescale): {ts[peaks] if len(peaks)>0 else 'none detected'}")

    mode_verdict = "MULTI-MODAL (≥3 modes) — PROCEED" if n_modes >= 3 else f"BI-MODAL ({n_modes} mode) — KILL: reduces to dual-rate"
    print(f"  Verdict: {mode_verdict}")

    # 6. If multi-modal, test band skip configs
    if n_modes >= 3:
        print("\n[5] Testing band-skip configs (L=2 vs L=3)...")

        # L=2 results (baseline: dual-rate equivalents)
        l2_results = test_all_band_configs(rssm, data, timescales_arr, BAND_CONFIGS_L2, L_label=2)

        # L=3 results (spectral)
        l3_results = test_all_band_configs(rssm, data, timescales_arr, BAND_CONFIGS_L3, L_label=3)

        # Pareto comparison
        print("\n[6] Pareto comparison: L=3 vs best L=2...")
        best_l2_quality = [r for r in l2_results if r['quality_pass']]
        best_l3_quality = [r for r in l3_results if r['quality_pass']]

        if best_l2_quality and best_l3_quality:
            best_l2_speed = max(r['theoretical_speedup'] for r in best_l2_quality)
            best_l3_speed = max(r['theoretical_speedup'] for r in best_l3_quality)
            pareto_advantage = best_l3_speed > best_l2_speed
            print(f"  Best L=2 speedup (quality pass): {best_l2_speed:.3f}x")
            print(f"  Best L=3 speedup (quality pass): {best_l3_speed:.3f}x")
            print(f"  L=3 Pareto advantage over L=2: {'YES ✓' if pareto_advantage else 'NO ✗'}")
        else:
            pareto_advantage = False
            print("  No quality-passing config found")

        overall_verdict = "CONDITIONAL-GO" if (n_modes >= 3 and any(r['speed_pass'] and r['quality_pass'] for r in l3_results)) else "FAIL"
    else:
        l2_results = []
        l3_results = []
        pareto_advantage = False
        overall_verdict = "KILL"

    # Save results
    results = {
        "idea": "spectral-multilevel-rssm",
        "kill_criterion_met": n_modes >= 3,
        "n_spectral_modes": n_modes,
        "mode_verdict": mode_verdict,
        "timescale_stats": {
            "min": float(timescales_arr.min()),
            "max": float(timescales_arr.max()),
            "mean": float(timescales_arr.mean()),
            "std": float(timescales_arr.std()),
        },
        "l2_results": l2_results,
        "l3_results": l3_results,
        "pareto_advantage_over_l2": pareto_advantage,
        "verdict": overall_verdict,
        "device": DEVICE,
    }

    out_path = Path(__file__).parent / "poc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== VERDICT: {overall_verdict} ===")
    print(f"Results saved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k not in ['l2_results', 'l3_results']}, indent=2))
