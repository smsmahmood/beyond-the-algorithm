"""Offline pipeline: raw Deezer features -> small, deployment-ready result tables.

Usage (from the project root):
    python scripts/prepare_data.py

Steps
1. Validate both CSVs (shape, columns, missing values, segment ids).
2. Draw R disjoint, seeded random samples of users, streaming the 1.8 GB user
   file in chunks so the whole file is never held in memory.
3. For each sample, compute modeled stream probabilities with Deezer's
   published model and evaluate every strategy on the same users and the
   same 12 slots.
4. Write tidy CSVs + meta.json to data/processed/ for the Streamlit app.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis as a  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SEED = 42
N_REPLICATES = 5
USERS_PER_REPLICATE = 50_000
CHUNK_ROWS = 100_000
LAMBDAS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
LOSS_BUDGETS = {"Balanced discovery": 2.0, "Diversity-first": 15.0}
ORDER_LAMBDAS = [0.1, 0.3, 0.5, 1.0]   # serving-order sensitivity check (replicate 1 only)
N_ORDERS = 5

N_DIMS = 96
N_SEGMENTS = 100


def count_rows(path: Path) -> int:
    """Count data rows by scanning newlines in binary (fast, constant memory)."""
    n = 0
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            n += block.count(b"\n")
    return n - 1  # header


def load_playlists() -> np.ndarray:
    df = pd.read_csv(RAW / "playlist_features.csv")
    expected = [f"dim_{i}" for i in range(N_DIMS)] + [str(N_DIMS)]
    assert list(df.columns) == expected, "unexpected playlist columns"
    assert df.notna().all().all(), "missing values in playlist features"
    return df.to_numpy(dtype=np.float64)


def sample_users(n_rows: int) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Stream the user file once and keep only the sampled rows.

    Returns per-replicate feature matrices (with the bias column of ones that
    Deezer's code appends), per-replicate k-means segment ids, and validation
    statistics computed over the *full* file.
    """
    rng = np.random.default_rng(SEED)
    total = N_REPLICATES * USERS_PER_REPLICATE
    order = rng.permutation(n_rows)[:total]          # random order == serving order
    position = np.full(n_rows, -1, dtype=np.int64)
    position[order] = np.arange(total)

    X = np.empty((total, N_DIMS + 1), dtype=np.float32)
    X[:, N_DIMS] = 1.0                                 # bias term, as in carousel_bandits/main.py
    seg = np.empty(total, dtype=np.int16)

    expected = ["segment"] + [f"dim_{i}" for i in range(N_DIMS)]
    dtypes = {"segment": np.int16, **{c: np.float32 for c in expected[1:]}}
    seg_counts = np.zeros(N_SEGMENTS, dtype=np.int64)
    n_missing, seen = 0, 0

    reader = pd.read_csv(RAW / "user_features.csv", dtype=dtypes, chunksize=CHUNK_ROWS)
    for chunk in reader:
        if seen == 0:
            assert list(chunk.columns) == expected, "unexpected user columns"
        n_missing += int(chunk.isna().sum().sum())
        segs = chunk["segment"].to_numpy()
        assert segs.min() >= 0 and segs.max() < N_SEGMENTS, "segment id out of range"
        seg_counts += np.bincount(segs, minlength=N_SEGMENTS)

        pos = position[seen:seen + len(chunk)]
        keep = pos >= 0
        X[pos[keep], :N_DIMS] = chunk.iloc[:, 1:].to_numpy()[keep]
        seg[pos[keep]] = segs[keep]
        seen += len(chunk)
        print(f"  read {seen:,}/{n_rows:,} rows", end="\r", flush=True)
    print()

    assert seen == n_rows, "row count mismatch"
    assert n_missing == 0, f"{n_missing} missing values in user features"

    splits = np.split(np.arange(total), N_REPLICATES)
    validation = {
        "n_users_file": int(n_rows),
        "missing_values": n_missing,
        "n_segments": int((seg_counts > 0).sum()),
        "segment_size_min": int(seg_counts.min()),
        "segment_size_max": int(seg_counts.max()),
    }
    return [X[i] for i in splits], [seg[i] for i in splits], validation


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Validating playlist features ...")
    W = load_playlists()
    print(f"  {W.shape[0]} playlists x {W.shape[1]} weights (incl. bias)")

    print("Counting user rows ...")
    n_rows = count_rows(RAW / "user_features.csv")
    print(f"  {n_rows:,} users")

    print(f"Sampling {N_REPLICATES} x {USERS_PER_REPLICATE:,} users (seed={SEED}) ...")
    Xs, segs, validation = sample_users(n_rows)

    results = {k: [] for k in ("curve", "tiers", "clusters", "exposure")}
    prob_stats = []
    for r, (X, seg) in enumerate(zip(Xs, segs)):
        print(f"Replicate {r + 1}/{N_REPLICATES}: scoring and allocating ...")
        probs = a.stream_probabilities(X.astype(np.float64), W)
        prob_stats.append(np.percentile(probs, [1, 50, 99, 99.9]))
        out = a.run_replicate(probs, seg, LAMBDAS, seed=SEED + r, keep_exposure=r == 0)
        if r == 0:
            print(f"  serving-order sensitivity ({N_ORDERS} orders) ...")
            order = a.order_sensitivity(probs, ORDER_LAMBDAS, N_ORDERS, seed=SEED)
        for k, df in out.items():
            if k == "exposure" and r > 0:
                continue  # one replicate is enough for per-playlist plots
            df.insert(0, "replicate", r)
            results[k].append(df)

    curve = pd.concat(results["curve"], ignore_index=True)
    tiers = pd.concat(results["tiers"], ignore_index=True)
    clusters = pd.concat(results["clusters"], ignore_index=True)
    exposure = pd.concat(results["exposure"], ignore_index=True)

    # Cluster results pooled across replicates (each cluster has ~2.5k sampled users).
    clusters = (clusters.groupby(["strategy", "lambda", "cluster"], dropna=False)
                [["n_users", "expected_reward", "baseline_reward"]].sum().reset_index())
    clusters["expected_reward"] /= clusters["n_users"]
    clusters["baseline_reward"] /= clusters["n_users"]
    clusters["reward_change_pct"] = 100 * (clusters["expected_reward"] / clusters["baseline_reward"] - 1)

    mean_curve = (curve[curve["strategy"] == "Exposure-aware"]
                  .groupby("lambda")["reward_change_pct"].mean().reset_index())
    presets = {"Engagement-first": 0.0}
    for name, budget in LOSS_BUDGETS.items():
        presets[name] = a.pick_by_loss_budget(mean_curve, budget)

    curve.to_csv(OUT / "curve.csv", index=False)
    tiers.to_csv(OUT / "tiers.csv", index=False)
    clusters.to_csv(OUT / "clusters.csv", index=False)
    exposure.to_csv(OUT / "exposure.csv", index=False)
    order.to_csv(OUT / "order_sensitivity.csv", index=False)

    p = np.mean(prob_stats, axis=0)
    meta = {
        "seed": SEED,
        "n_replicates": N_REPLICATES,
        "users_per_replicate": USERS_PER_REPLICATE,
        "n_playlists": int(W.shape[0]),
        "n_slots": a.N_SLOTS,
        "l_init": a.L_INIT,
        "lambdas": LAMBDAS,
        "loss_budgets_pct": LOSS_BUDGETS,
        "presets": presets,
        "validation": {
            **validation,
            "playlist_bias_mean": float(W[:, -1].mean()),
            "stream_prob_p1": float(p[0]),
            "stream_prob_median": float(p[1]),
            "stream_prob_p99": float(p[2]),
            "stream_prob_p99_9": float(p[3]),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Presets: {presets}")
    print(f"Done in {meta['runtime_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
