"""Strategy comparison: run every strategy on the same users and summarize.

Everything here operates on one replicate (one reproducible user sample).
`scripts/prepare_data.py` runs several replicates and writes the results the
dashboard reads, so the app never touches the raw 1.8 GB file.

Exposure is measured two ways:
- placements: all 12 carousel slots (what the strategy assigns)
- first-view placements: slots 1-3, the initially visible positions
Both count what the ranker assigns. The data has no impression or scroll
records, so actual viewed exposure is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as m
from src import strategies as s

N_SLOTS = 12   # carousel length (L in Bendada et al., 2020)
L_INIT = 3     # cards visible before scrolling (L_init)
N_TIERS = 4

TIER_LABELS = {
    0: "Least likely to stream",
    1: "Below median",
    2: "Above median",
    3: "Most likely to stream",
}


def stream_probabilities(user_X: np.ndarray, playlist_W: np.ndarray) -> np.ndarray:
    """Deezer's ground-truth model: sigmoid(user features incl. bias . playlist weights)."""
    return m.sigmoid(user_X @ playlist_W.T).astype(np.float32)


def evaluate(probs: np.ndarray, recos: np.ndarray) -> dict:
    """Per-user modeled engagement and catalog-level placements for one set of carousels."""
    slot_p = np.take_along_axis(probs, recos, axis=1).astype(np.float64)
    n_items = probs.shape[1]
    return {
        "user_reward": m.slate_stream_probability(slot_p),
        "user_initial": m.initial_view_stream_probability(slot_p, L_INIT),
        "user_slot_mean": slot_p.mean(axis=1),
        "placements": m.placement_counts(recos, n_items),
        "first_view_placements": m.placement_counts(recos[:, :L_INIT], n_items),
    }


def summarize(ev: dict) -> dict:
    """Population-level engagement and exposure metrics for one strategy."""
    pl, fv = ev["placements"], ev["first_view_placements"]
    return {
        "n_users": int(ev["user_reward"].shape[0]),
        "expected_reward": float(ev["user_reward"].mean()),
        "initial_view_reward": float(ev["user_initial"].mean()),
        "slot_stream_rate": float(ev["user_slot_mean"].mean()),
        "coverage": m.catalog_coverage(pl),
        "effective_catalog": m.effective_catalog_size(pl),
        "gini": m.gini(pl),
        "top10_share": m.top_share(pl, 0.10),
        "fv_coverage": m.catalog_coverage(fv),
        "fv_effective_catalog": m.effective_catalog_size(fv),
    }


def propensity_tiers(baseline_reward: np.ndarray, n_tiers: int = N_TIERS) -> np.ndarray:
    """Quartiles of each user's engagement-first expected reward (0 = lowest)."""
    edges = np.quantile(baseline_reward, np.linspace(0, 1, n_tiers + 1)[1:-1])
    return np.searchsorted(edges, baseline_reward, side="right")


def run_replicate(
    probs: np.ndarray,
    clusters: np.ndarray,
    lambdas: list[float],
    seed: int,
    keep_exposure: bool = False,
) -> dict[str, pd.DataFrame]:
    """Evaluate the engagement-first baseline, the exposure-aware grid and a random reference.

    Returns tidy DataFrames: overall curve, propensity-tier results,
    k-means-cluster results and (optionally) per-playlist placement counts.
    """
    n_users, n_items = probs.shape
    base = evaluate(probs, s.engagement_first(probs, N_SLOTS))
    tiers = propensity_tiers(base["user_reward"])

    curve, tier_rows, cluster_rows, exposure_rows = [], [], [], []

    def record(label: str, lam: float, recos: np.ndarray, ev: dict):
        curve.append({"strategy": label, "lambda": lam, **summarize(ev)})
        for t in range(N_TIERS):
            mask = tiers == t
            tier_pl = m.placement_counts(recos[mask], n_items)
            tier_rows.append({
                "strategy": label, "lambda": lam, "tier": t,
                "n_users": int(mask.sum()),
                "expected_reward": float(ev["user_reward"][mask].mean()),
                "baseline_reward": float(base["user_reward"][mask].mean()),
                "initial_view_reward": float(ev["user_initial"][mask].mean()),
                "baseline_initial_view_reward": float(base["user_initial"][mask].mean()),
                "coverage": m.catalog_coverage(tier_pl),
                "effective_catalog": m.effective_catalog_size(tier_pl),
            })
        cl = pd.DataFrame({"cluster": clusters, "r": ev["user_reward"], "b": base["user_reward"]})
        agg = cl.groupby("cluster").agg(n_users=("r", "size"), expected_reward=("r", "sum"),
                                        baseline_reward=("b", "sum")).reset_index()
        agg["strategy"], agg["lambda"] = label, lam
        cluster_rows.append(agg)
        if keep_exposure:
            exposure_rows.append(pd.DataFrame({
                "strategy": label, "lambda": lam, "playlist": np.arange(n_items),
                "placements": ev["placements"], "first_view_placements": ev["first_view_placements"],
            }))

    for lam in lambdas:
        recos = s.exposure_aware(probs, N_SLOTS, lam)
        record("Exposure-aware", lam, recos, evaluate(probs, recos))

    rnd = s.random_baseline(n_users, n_items, N_SLOTS, seed)
    record("Random", np.nan, rnd, evaluate(probs, rnd))

    curve = pd.DataFrame(curve)
    base_reward = curve.loc[curve["lambda"] == 0, "expected_reward"].iloc[0]
    curve["reward_change_pct"] = m.relative_change(curve["expected_reward"], base_reward)

    return {
        "curve": curve,
        "tiers": pd.DataFrame(tier_rows),
        "clusters": pd.concat(cluster_rows, ignore_index=True),
        "exposure": pd.concat(exposure_rows, ignore_index=True) if exposure_rows else pd.DataFrame(),
    }


def order_sensitivity(probs: np.ndarray, lambdas: list[float], n_orders: int, seed: int) -> pd.DataFrame:
    """Re-run the exposure-aware ranker on the same users in different serving orders.

    The ranker is sequential (early users see less-penalized scores), so its
    output could depend on who is served first. Order 0 is the original order.
    Also records the engagement change for the first and last 10% of users served,
    which shows whether serving position itself drives individual outcomes.
    """
    rng = np.random.default_rng(seed)
    n_users = probs.shape[0]
    base = evaluate(probs, s.engagement_first(probs, N_SLOTS))
    decile = n_users // 10
    rows = []
    for o in range(n_orders):
        perm = np.arange(n_users) if o == 0 else rng.permutation(n_users)
        for lam in lambdas:
            recos = np.empty((n_users, N_SLOTS), dtype=np.int64)
            recos[perm] = s.exposure_aware(probs[perm], N_SLOTS, lam)
            ev = evaluate(probs, recos)
            first, last = perm[:decile], perm[-decile:]
            rows.append({
                "order": o, "lambda": lam, **summarize(ev),
                "reward_change_pct": m.relative_change(ev["user_reward"].mean(), base["user_reward"].mean()),
                "first_decile_change_pct": m.relative_change(ev["user_reward"][first].mean(),
                                                             base["user_reward"][first].mean()),
                "last_decile_change_pct": m.relative_change(ev["user_reward"][last].mean(),
                                                            base["user_reward"][last].mean()),
            })
    return pd.DataFrame(rows)


def pick_by_loss_budget(curve: pd.DataFrame, max_loss_pct: float) -> float:
    """Largest lambda whose mean modeled-engagement loss stays within the budget.

    `curve` must hold one row per lambda for the exposure-aware strategy with a
    `reward_change_pct` column (negative = loss). Coverage rises monotonically
    with lambda in our grid, so the largest admissible lambda is also the most
    diverse admissible option.
    """
    ok = curve[(curve["reward_change_pct"] >= -max_loss_pct)]
    return float(ok["lambda"].max())
