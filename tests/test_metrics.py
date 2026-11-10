import numpy as np
import pytest

from src import metrics as m


def test_slate_probability_matches_deezer_formula():
    p = np.array([[0.1, 0.2, 0.5], [0.0, 0.0, 0.0]])
    expected = np.array([1 - 0.9 * 0.8 * 0.5, 0.0])
    np.testing.assert_allclose(m.slate_stream_probability(p), expected)


def test_initial_view_uses_only_first_slots():
    p = np.array([[0.5, 0.0, 0.0, 0.9]])
    assert m.initial_view_stream_probability(p, l_init=3)[0] == pytest.approx(0.5)


def test_relative_change():
    assert m.relative_change(0.9, 1.0) == pytest.approx(-10.0)


def test_coverage_and_impressions():
    recos = np.array([[0, 1], [0, 2]])
    imp = m.placement_counts(recos, n_items=5)
    np.testing.assert_array_equal(imp, [2, 1, 1, 0, 0])
    assert m.catalog_coverage(imp) == pytest.approx(0.6)


def test_effective_catalog_size_bounds():
    assert m.effective_catalog_size(np.full(10, 7)) == pytest.approx(10)
    assert m.effective_catalog_size(np.array([5, 0, 0])) == pytest.approx(1)


def test_effective_catalog_distinguishes_spread_at_equal_coverage():
    # Both cover 2 of 2 items, but one is far more concentrated.
    even, skewed = np.array([50, 50]), np.array([99, 1])
    assert m.catalog_coverage(even) == m.catalog_coverage(skewed)
    assert m.effective_catalog_size(even) > m.effective_catalog_size(skewed)


def test_gini_bounds():
    assert m.gini(np.full(10, 3)) == pytest.approx(0.0)
    assert m.gini(np.array([0, 0, 0, 10])) == pytest.approx(0.75)


def test_top_share():
    imp = np.array([60, 10, 10, 10, 10, 0, 0, 0, 0, 0])
    assert m.top_share(imp, 0.10) == pytest.approx(0.6)


def test_ni_sample_size_reference_value():
    # Hand calculation: (1.95996 + 0.84162)^2 * 2 * 0.3 * 0.7 / (0.02 * 0.3)^2 = 91,571
    assert m.ni_sample_size_per_arm(0.30, 0.02) == 91_571


def test_ni_sample_size_grows_as_effect_nears_margin():
    sizes = [m.ni_sample_size_per_arm(0.30, 0.02, e) for e in (0.0, -0.005, -0.01, -0.015)]
    assert sizes == sorted(sizes)
    assert m.ni_sample_size_per_arm(0.30, 0.02, -0.02) == float("inf")


def test_ni_sample_size_achieves_nominal_power_by_simulation():
    """Simulate the one-sided Wald non-inferiority test at the computed n."""
    rng = np.random.default_rng(7)
    p_c, margin, effect = 0.30, 0.10, -0.03       # large margin keeps n small for a fast test
    n = int(m.ni_sample_size_per_arm(p_c, margin, effect, alpha=0.025, power=0.8))
    p_t = p_c * (1 + effect)
    sims = 20_000
    xc, xt = rng.binomial(n, p_c, sims) / n, rng.binomial(n, p_t, sims) / n
    se = np.sqrt(xc * (1 - xc) / n + xt * (1 - xt) / n)
    reject = (xt - xc + margin * p_c) / se > 1.959964
    assert reject.mean() == pytest.approx(0.80, abs=0.02)

    # Type I error at the margin boundary stays near one-sided alpha.
    p_t0 = p_c * (1 - margin)
    xt0 = rng.binomial(n, p_t0, sims) / n
    se0 = np.sqrt(xc * (1 - xc) / n + xt0 * (1 - xt0) / n)
    assert ((xt0 - xc + margin * p_c) / se0 > 1.959964).mean() == pytest.approx(0.025, abs=0.006)
