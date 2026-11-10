import numpy as np
import pytest

from src import analysis as a
from src import metrics as m
from src import strategies as s


@pytest.fixture(scope="module")
def probs():
    rng = np.random.default_rng(0)
    users, items = rng.normal(size=(2000, 8)), rng.normal(size=(60, 8))
    # A shared popularity term concentrates engagement-first exposure, as in the real data.
    return m.sigmoid(users @ items.T + np.linspace(1, -4, 60)).astype(np.float32)


def test_engagement_first_is_per_user_top_k(probs):
    k = 5
    recos = s.engagement_first(probs, k)
    expected = np.sort(np.argsort(-probs, axis=1)[:, :k], axis=1)
    np.testing.assert_array_equal(np.sort(recos, axis=1), expected)
    # Displayed best-first.
    slot_p = np.take_along_axis(probs, recos, axis=1)
    assert (np.diff(slot_p, axis=1) <= 0).all()


def test_lambda_zero_equals_engagement_first(probs):
    np.testing.assert_array_equal(s.exposure_aware(probs, 5, 0.0), s.engagement_first(probs, 5))


@pytest.mark.parametrize("lam", [0.3, 1.0, 3.0])
def test_exposure_aware_returns_k_distinct_items(probs, lam):
    recos = s.exposure_aware(probs, 5, lam)
    assert recos.shape == (probs.shape[0], 5)
    assert all(len(set(r)) == 5 for r in recos)


def test_tradeoff_direction(probs):
    """More diversity weight -> lower modeled engagement, broader exposure."""
    rewards, eff = [], []
    for lam in [0.0, 0.5, 2.0]:
        recos = s.exposure_aware(probs, 5, lam)
        ev = a.evaluate(probs, recos)
        rewards.append(ev["user_reward"].mean())
        eff.append(m.effective_catalog_size(ev["placements"]))
    assert rewards[0] >= rewards[1] >= rewards[2]
    assert eff[0] < eff[1] < eff[2]


def test_every_strategy_shows_same_number_of_impressions(probs):
    n_users, n_items = probs.shape
    for recos in (s.engagement_first(probs, 5), s.exposure_aware(probs, 5, 1.0),
                  s.random_baseline(n_users, n_items, 5, seed=1)):
        assert m.placement_counts(recos, n_items).sum() == n_users * 5


def test_run_replicate_baseline_and_budget(probs):
    out = a.run_replicate(probs, np.zeros(len(probs), dtype=int), [0.0, 0.5, 2.0], seed=0)
    curve = out["curve"]
    assert curve.loc[curve["lambda"] == 0, "reward_change_pct"].iloc[0] == pytest.approx(0)
    assert len(out["tiers"]) == len(curve) * a.N_TIERS
    ea = curve[curve["strategy"] == "Exposure-aware"]
    assert a.pick_by_loss_budget(ea, 0.0) == 0.0
    assert a.pick_by_loss_budget(ea, 100.0) == 2.0


def test_first_view_placements_use_first_slots_only(probs):
    recos = s.engagement_first(probs, a.N_SLOTS)
    ev = a.evaluate(probs, recos)
    assert ev["first_view_placements"].sum() == probs.shape[0] * a.L_INIT
    np.testing.assert_array_equal(ev["first_view_placements"],
                                  m.placement_counts(recos[:, :a.L_INIT], probs.shape[1]))


def test_order_sensitivity_original_order_matches_main_run(probs):
    out = a.order_sensitivity(probs, [0.5], n_orders=2, seed=0)
    main = a.run_replicate(probs, np.zeros(len(probs), dtype=int), [0.0, 0.5], seed=0)["curve"]
    expected = main.loc[main["lambda"] == 0.5, "reward_change_pct"].iloc[0]
    assert out.loc[out["order"] == 0, "reward_change_pct"].iloc[0] == pytest.approx(expected)
    assert len(out) == 2
