"""Engagement and diversity metrics for carousel recommendations.

Engagement metrics are *modeled*: they apply Deezer's published ground-truth
stream model, p(u, i) = sigmoid(x_u . theta_i), to the playlists a strategy
recommends. They are expected values, not observed streams.

Diversity metrics describe how *placements* are spread across the playlist
catalog. A placement is one playlist assigned to one carousel slot for one
user. It is not a viewed impression: only the first L_init slots are visible
without scrolling, and the dataset contains no scroll behavior. Callers can
therefore compute the same metrics over all k slots ("placements") or over
the first L_init slots only ("first-view placements", the initially visible
positions). Neither is an observed view. Every strategy produces exactly n_users * slots placements,
so the metrics compare *how* placements are distributed, not how many cards
are shown.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Engagement (modeled)
# ---------------------------------------------------------------------------

def slate_stream_probability(slot_probs: np.ndarray) -> np.ndarray:
    """P(user streams at least one playlist in the slate).

    Matches `compute_theoretical_rewards` in Deezer's carousel_bandits
    environment: 1 - prod_i (1 - p_i), assuming independent stream decisions
    across displayed playlists.

    slot_probs: (n_users, n_slots) stream probabilities of the displayed playlists.
    Returns: (n_users,) expected reward per user.
    """
    slot_probs = np.asarray(slot_probs, dtype=np.float64)
    return 1.0 - np.prod(1.0 - slot_probs, axis=1)


def initial_view_stream_probability(slot_probs: np.ndarray, l_init: int = 3) -> np.ndarray:
    """P(stream among the first `l_init` cards, which are visible without scrolling).

    Assumes slots are ordered as displayed (left to right). L_init = 3 follows
    the carousel setup in Bendada et al. (2020).
    """
    return slate_stream_probability(np.asarray(slot_probs)[:, :l_init])


def relative_change(value: float, baseline: float) -> float:
    """Percent change of `value` relative to `baseline`."""
    return 100.0 * (value - baseline) / baseline


# ---------------------------------------------------------------------------
# Diversity / exposure distribution
# ---------------------------------------------------------------------------

def placement_counts(recos: np.ndarray, n_items: int) -> np.ndarray:
    """Number of carousel slots each playlist occupies across all users."""
    return np.bincount(np.asarray(recos).ravel(), minlength=n_items).astype(np.int64)


def catalog_coverage(placements: np.ndarray) -> float:
    """Share of the catalog placed for at least one user (denominator: all playlists)."""
    placements = np.asarray(placements)
    return float((placements > 0).mean())


def effective_catalog_size(placements: np.ndarray) -> float:
    """exp(Shannon entropy) of the placement distribution.

    Reads as "the number of equally exposed playlists that would give the same
    spread". Equals the catalog size when exposure is uniform and 1 when a
    single playlist gets every placement.
    """
    placements = np.asarray(placements, dtype=np.float64)
    shares = placements[placements > 0] / placements.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))


def gini(placements: np.ndarray) -> float:
    """Gini coefficient of placements across all playlists (0 = equal, ->1 = concentrated)."""
    x = np.sort(np.asarray(placements, dtype=np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(((2 * ranks - n - 1) * x).sum() / (n * x.sum()))


def top_share(placements: np.ndarray, frac: float = 0.10) -> float:
    """Share of all placements captured by the most-exposed `frac` of the catalog."""
    x = np.sort(np.asarray(placements, dtype=np.float64))[::-1]
    n_top = max(1, int(round(frac * x.size)))
    return float(x[:n_top].sum() / x.sum())


# ---------------------------------------------------------------------------
# Experiment planning
# ---------------------------------------------------------------------------

def ni_sample_size_per_arm(
    p_control: float,
    margin_rel: float,
    effect_rel: float = 0.0,
    alpha: float = 0.025,
    power: float = 0.8,
) -> float:
    """Users per arm for a one-sided non-inferiority test of two proportions.

    H0: p_t - p_c <= -delta  vs  H1: p_t - p_c > -delta, with delta = margin_rel * p_control.
    Wald (unpooled) normal approximation with equal allocation:

        n = (z_{1-alpha} + z_{1-beta})^2 [p_c(1-p_c) + p_t(1-p_t)] / (p_t - p_c + delta)^2

    margin_rel and effect_rel are relative fractions (0.02 = 2%). Returns inf when
    the assumed true effect is at or beyond the margin (the test cannot be powered).
    """
    from statistics import NormalDist

    p_t = p_control * (1 + effect_rel)
    gap = (p_t - p_control) + margin_rel * p_control
    if gap <= 0:
        return float("inf")
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    variance = p_control * (1 - p_control) + p_t * (1 - p_t)
    return float(np.ceil(z ** 2 * variance / gap ** 2))
