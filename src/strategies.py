"""Recommendation strategies for a k-slot playlist carousel.

All strategies receive the same matrix of modeled stream probabilities
(users x playlists) and return, for each user, k distinct playlist indices in
display order (slot 0 = leftmost card).

Engagement-first
    Each user gets the k playlists with the highest modeled stream probability.

Exposure-aware allocation (balanced / diversity-first)
    Our own diversity-aware re-ranking. Users are served sequentially in small
    batches (random order). Before each batch, every playlist's score is
    discounted by how over-exposed it already is relative to a perfectly even
    allocation:

        score(u, i) = log p(u, i) - lambda * log(1 + E_i / E_bar)

    E_i is the number of placements playlist i has received so far, and
    E_bar = (users served so far * k) / n_playlists is the even-share benchmark.
    Equivalently, a playlist's probability is divided by (1 + E_i/E_bar)^lambda.

    lambda = 0 reproduces engagement-first. Larger lambda trades modeled
    engagement for a flatter exposure distribution. Because the penalty is
    multiplicative in p, it is comparable across users with very different
    baseline stream propensities. The penalty depends only on running exposure
    counts, so it could be served online with a rolling counter.

    Within the selected k, cards are ordered by the user's own p(u, i)
    (highest first), so diversification changes *what* is shown, while each
    carousel still leads with its strongest options.
"""

from __future__ import annotations

import numpy as np


def _top_k_ordered(scores: np.ndarray, k: int, order_by: np.ndarray) -> np.ndarray:
    """Pick top-k columns by `scores`, then order them by `order_by` (descending)."""
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    key = np.take_along_axis(order_by, top, axis=1)
    return np.take_along_axis(top, np.argsort(-key, axis=1), axis=1)


def engagement_first(probs: np.ndarray, k: int) -> np.ndarray:
    """Top-k playlists per user by modeled stream probability."""
    return _top_k_ordered(probs, k, probs)


def exposure_aware(
    probs: np.ndarray,
    k: int,
    lam: float,
    batch_size: int = 250,
) -> np.ndarray:
    """Exposure-aware sequential allocation (see module docstring).

    Users are served in row order, so callers should pass users in a random,
    seeded order.
    """
    if lam == 0:
        return engagement_first(probs, k)

    n_users, n_items = probs.shape
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    exposure = np.zeros(n_items, dtype=np.float64)
    recos = np.empty((n_users, k), dtype=np.int64)

    for start in range(0, n_users, batch_size):
        stop = min(start + batch_size, n_users)
        even_share = start * k / n_items
        if even_share > 0:
            penalty = lam * np.log1p(exposure / even_share)
            scores = log_p[start:stop] - penalty
        else:
            scores = log_p[start:stop]
        batch = _top_k_ordered(scores, k, probs[start:stop])
        recos[start:stop] = batch
        exposure += np.bincount(batch.ravel(), minlength=n_items)

    return recos


def random_baseline(n_users: int, n_items: int, k: int, seed: int) -> np.ndarray:
    """Uniformly random k playlists per user. A reference point, not a candidate strategy."""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_users, n_items)), axis=1)[:, :k]
