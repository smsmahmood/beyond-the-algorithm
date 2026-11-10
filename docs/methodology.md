# Methodology

This document records the analytical choices behind **Beyond the Algorithm**: what the data is, how engagement
is modeled, how the strategies work, how each metric is defined, and what the results can and cannot support.

## 1. Data

| File | Shape | Contents |
|---|---|---|
| `user_features.csv` | 974,960 × 97 | `segment` (k-means cluster id, 0–99) and `dim_0`…`dim_95`, an anonymized 96-dim user embedding |
| `playlist_features.csv` | 862 × 97 | `dim_0`…`dim_95` plus column `96`, a 97-dim weight vector θᵢ per playlist |

- **Source:** Bendada, Salha & Bontempelli (2020), *Carousel Personalization in Music Streaming Apps with
  Contextual Bandits*, RecSys 2020. Data on Zenodo, DOI [10.5281/zenodo.4048678](https://zenodo.org/records/4048678).
  Code: [deezer/carousel_bandits](https://github.com/deezer/carousel_bandits).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution is given in the README, the
  app and here. The authors ask users of the data to cite the paper.
- **What is observed vs. modeled:** the embeddings and weights are real, anonymized artifacts derived from Deezer
  usage data. Stream outcomes are **not** in the dataset. They come from the model below. Nothing here is an
  outcome of Deezer's live experiments.

### Validation (run by `scripts/prepare_data.py`)
- Exact column names and order checked for both files.
- 974,960 user rows. The count from a binary newline scan matches the rows streamed through pandas.
- 0 missing values in either file.
- 100 segment ids, all within 0–99 (cluster sizes 6,607–45,975).
- Playlist column `96` behaves as an intercept (mean −5.42). This matches the bias term Deezer's code appends to
  user vectors as a constant 1.

## 2. Engagement model (Deezer's published simulator)

From `environment.py` / `main.py` in `deezer/carousel_bandits`:

- User vector xᵤ = [96-dim embedding, 1].
- Probability that user u streams playlist i when it is displayed: **p(u,i) = σ(xᵤ · θᵢ)**.
- Expected reward of a slate S (Deezer's `compute_theoretical_rewards`): **1 − ∏ᵢ∈S (1 − p(u,i))**, the probability
  of at least one stream, assuming independent decisions across displayed cards.

We use this model as published, with no new probability formula. We compute **expected values**, so results
contain no Monte Carlo noise. Deezer's `simulate_batch_users_reward` additionally draws Bernoulli outcomes and
applies a cascade (only the first stream counts). Its expectation for a single display is the same quantity.

On our samples, the median p(u,i) is about 0.5% and the 99.9th percentile about 44%. Probability mass is
concentrated on a few user–playlist pairs, which is why engagement-first exposure is so concentrated.

## 3. Sampling and reproducibility

- Seed 42. A single random permutation of all 974,960 row indices yields **5 disjoint samples of 50,000 users**.
- The 1.8 GB file is streamed in 100k-row chunks (float32), and only sampled rows are kept (~100 MB in memory).
- Within a sample, the permutation order is also the **serving order** for the exposure-aware ranker.
- All strategies are evaluated on the same users with the same 12 slots (L = 12, L_init = 3, as in the paper).
- The full pipeline takes about 1 minute on a laptop. Re-running it reproduces `data/processed/` exactly.

## 4. Strategies

All strategies output 12 distinct playlists per user, ordered for display.

**Engagement-first.** The top 12 playlists by p(u,i), displayed in descending p.

**Exposure-aware ranker (our implementation).** Users are served in batches of 250. Before each batch:

```
score(u, i) = log p(u, i) − λ · log(1 + E_i / Ē)
```

- E_i is the number of placements playlist i has received so far.
- Ē = (users served so far × 12) / 862 is the placements each playlist would have under a perfectly even
  allocation.

Each user gets the top 12 by score, displayed in descending p(u,i).

- λ = 0 reproduces engagement-first exactly (unit-tested).
- The penalty is multiplicative in p (a playlist at 3× its even share has its probability divided by 4^λ), so it is
  comparable across listeners whose baseline propensities differ by orders of magnitude.
- It only needs running exposure counters, so it could be served online.
- It is a transparent heuristic, not an optimal solver. A constrained optimization (e.g. maximize engagement
  subject to exposure floors) is a natural extension.

**Presets** are chosen by rule from the λ grid {0, 0.025, …, 5}, using engagement loss averaged over the 5 samples:

| Preset | Rule | λ |
|---|---|---|
| Engagement-first | λ = 0 | 0 |
| Balanced discovery | largest λ with mean modeled-engagement loss ≤ 2% | 0.3 |
| Diversity-first | largest λ with mean loss ≤ 15% | 1.0 |

The loss budgets are product choices, not statistical thresholds. The dashboard lets the reader move the budget.

**Random reference.** Twelve uniformly random playlists per user. It marks the bounds of the curve and is not a
candidate strategy.

## 5. Metric definitions

A **placement** is one playlist assigned to one carousel slot for one user. It is **not** a viewed impression.
Only the first 3 cards are visible without scrolling, and the dataset contains no scroll behavior. We therefore
report placements two ways: over all 12 slots and over slots 1–3, the initially visible positions. Both describe
what the ranker assigns. Neither is an observed view, stream or discovery. Actual viewed exposure is unknown and
would have to be measured in an online experiment.

| Metric (dashboard label) | Definition | Denominator |
|---|---|---|
| Predicted engagement | mean over users of 1 − ∏(1 − pᵢ) over the 12 placed playlists | users in sample (or group) |
| First-screen engagement | same over slots 1–3 | users |
| Change (%) / (pts) | relative / absolute difference vs engagement-first, same users | engagement-first value |
| Catalog placed (coverage) | share of playlists with ≥ 1 placement in any slot | 862 playlists |
| First-screen catalog | share of playlists with ≥ 1 placement in slots 1–3 | 862 playlists |
| Effective variety (effective catalog size) | exp(Shannon entropy of placement shares) | users × 12 placements |
| Gini | inequality of placements over all 862 playlists | 862 playlists |
| Top-10% share | share of placements going to the 86 most-placed playlists | users × 12 |

**Why more than coverage.** Every strategy makes exactly users × 12 placements, so no metric can be inflated by
showing more cards. Coverage only asks whether a playlist was placed at all. Effective variety and Gini measure how
evenly placements are spread. The unit tests include an example with equal coverage and different spread.

**Coverage depends on population size.** Coverage is computed per 50,000-user sample (or per ~12,500-user
quartile). With all 975k users, every strategy would reach more playlists. Compare strategies with each other,
not with an absolute target.

**What coverage is not.** These playlists are anonymized. Coverage is not genre diversity, artist diversity or
emerging-artist exposure, and we make no claim about those.

## 6. Segments

- **Likelihood-to-stream quartiles (primary).** Each user's engagement-first expected reward, cut into quartiles
  within each sample. This is a model-derived segment ("how likely is this listener to stream anything from the
  carousel"). It answers who bears the cost of diversification. Changes are reported in relative % and in
  percentage points.
- **Deezer k-means clusters (secondary).** The 100 provided clusters, pooled across samples (about 2,500 users
  each). They have no published meaning, so we only report the distribution of effects.

## 7. Key results (mean of 5 samples)

| λ | Predicted engagement | Δ (rel.) | Δ (pts) | First-screen Δ | Catalog placed | First-screen catalog | Effective variety |
|---|---|---|---|---|---|---|---|
| 0 (Engagement-first) | 74.3% | — | — | — | 10.6% | 5.5% | 26 |
| 0.1 | 74.3% | −0.12% | −0.09 | −0.00% | 14.1% | 5.5% | 30 |
| **0.3 (Balanced)** | 73.2% | **−1.51%** | **−1.12** | −0.12% | **23.0%** | 6.8% | **45** |
| 0.5 | 71.1% | −4.34% | −3.23 | −0.91% | 37.0% | 11.3% | 71 |
| **1.0 (Diversity-first)** | 65.0% | **−12.52%** | **−9.31** | −5.04% | **64.8%** | 25.8% | **181** |
| 2.0 | 42.5% | −42.84% | −31.85 | −39.39% | 92.5% | 48.2% | 505 |

**Most added variety at the balanced setting lands in slots 4–12.** Catalog placed doubles, but the first-screen
catalog moves only from 5.5% to 6.8%. The low engagement cost and the limited first-screen change have the same
cause: each carousel still leads with the listener's strongest options. Under balanced discovery, 6.8% of the
catalog is placed at least once in the first three positions and 23.0% at least once across all 12. How much of
either is actually viewed is unknown.

Engagement change by likelihood-to-stream quartile:

| λ | Q1 (least likely) | Q2 | Q3 | Q4 (most likely) |
|---|---|---|---|---|
| Baseline engagement | 41.1% | 68.8% | 89.4% | 98.1% |
| 0.3 | −7.7% (−3.16 pts) | −1.6% (−1.13 pts) | −0.2% (−0.18 pts) | −0.02% (−0.02 pts) |
| 1.0 | −47.3% (−19.4 pts) | −17.9% (−12.3 pts) | −5.3% (−4.7 pts) | −0.8% (−0.8 pts) |

The cost of diversification is regressive in both relative and absolute terms. The top quartile's near-zero change
is partly mechanical: the 1 − ∏ metric saturates near 1. Our likely explanation for Q1 is that these listeners
depend on a few popular matches, which are the playlists the ranker discounts. We have not verified that mechanism
directly.

## 8. Robustness and uncertainty

The model is deterministic, so variation comes only from which users are sampled and the order they are served in.

| Check | Balanced (λ = 0.3) | Diversity-first (λ = 1.0) |
|---|---|---|
| 5 disjoint user samples: engagement change | −1.51% (range < 0.01 pts) | −12.43% to −12.55% |
| 5 serving orders, same 50k users: engagement change | −1.505% to −1.513% | −12.53% to −12.57% |
| 5 serving orders: catalog placed | 22.4%–22.7% | 65.0%–65.4% |
| First vs last 10% of users served | −1.65% vs −1.48% | −17.8% vs −12.0% |

- **Aggregate results do not depend on serving order.**
- **Individual outcomes do, at high λ.** Users served early are measured against a noisy even-share benchmark built
  from few placements. At λ = 1 they lose about 6 pts more than users served late. At the balanced setting the gap
  is small (0.17 pts). A live system would warm-start exposure counters from recent traffic.

These ranges show that the **calculation is stable**. They are **not** confidence intervals for a real-world
treatment effect. The real effect could differ from the modeled one by far more than these ranges, because of
model error.

## 9. Measured results vs. hypotheses

| Statement | Status |
|---|---|
| Predicted engagement change per strategy and listener group | Measured (model-based) |
| Catalog placed, first-screen catalog, effective variety | Measured (on simulated carousels) |
| Stability across samples and serving orders | Measured |
| Real listeners respond as the model predicts | Hypothesis: requires the A/B test |
| Under-exposed playlists stay unlearned (feedback loop) | Hypothesis: not modeled |
| Broader exposure benefits creators or catalog health | Hypothesis: playlists are anonymized |
| Discovery improves satisfaction or retention | Hypothesis: no long-term outcomes in data |
| Extra exposure has learning value for future models | Hypothesis: static analysis |
| Catalog placed = genre or artist diversity | Not supported |

## 10. Offline evaluation vs. online experimentation

| | This analysis | An online A/B test |
|---|---|---|
| Question | If the model is right, what does each policy cost and gain? | What happens to real listeners? |
| Outcome | Modeled expected streams | Observed streams, listening time, retention |
| Main risk | Model misspecification, especially for rarely shown playlists | Novelty effects, interference, power |
| Causal claim | None | Yes, for the tested population and period |

### Proposed test and sample size

- **Primary metric:** whether a randomized user streams at least one playlist from their **first eligible
  recommendation carousel** within a predefined observation window (illustratively **30 minutes** after the
  render). One binary outcome per user. Using the first carousel only keeps the unit close to the offline
  single-carousel evaluation and avoids counting outcomes after users have already experienced the treatment.
  It is still **not equivalent** to the offline modeled probability, which assumes all 12 cards are seen and comes
  from a model.
- **Assignment vs. exposure.** Users are randomized (assigned) on entry. An *eligible exposure* is the first time
  during a 14-day enrollment window (illustrative) that the home screen renders the carousel. This render is logged
  identically in both arms before ranking is applied, so the arm cannot change who becomes eligible.
- **Users without an eligible exposure** are excluded from the primary (triggered) analysis. Because eligibility is
  arm-independent and precedes any treatment content, this does not create post-randomization selection bias.
  Viewing, scrolling or clicking must never be used as a filter, because the treatment can affect them. Checks:
  SRM on eligible users and pre-period covariate balance among eligible users. Sensitivity: an intent-to-treat
  analysis over all assigned users, counting non-eligible users as 0.
- **Hypothesis:** one-sided non-inferiority. H₀: p_t − p_c ≤ −δ vs H₁: p_t − p_c > −δ, with δ = 2% of p_c,
  α = 0.025 (one-sided). The margin mirrors the offline loss budget used to choose λ, but it is a business choice
  on a different metric and should be confirmed with stakeholders.
- **Sample size** (Wald approximation, independent proportions):
  n per arm = (z₁₋α + z₁₋β)² [p_c(1 − p_c) + p_t(1 − p_t)] / (p_t − p_c + δ)².
  This formula still fits the revised metric: one binary outcome per independently randomized user, compared as
  a difference in proportions. n counts *eligible* users per arm. Users to enroll = n ÷ expected eligibility rate.
  The control rate (default 0.30) is an illustrative assumption, to be replaced with the observed first-carousel
  rate from pre-period logs. The assumed true effect can be
  set to 0, the first-screen model prediction (−0.12%) or the all-12-card prediction (−1.51%). With the latter,
  the expected effect sits close to the 2% margin and the required sample grows roughly 16-fold. That is a real
  planning risk, which the dashboard makes visible.
- **Sensitivity to the margin.** n scales with 1 / (distance between the true effect and the margin)². With
  p_c = 0.30 and 80% power: no effect needs about 92k users per arm, an effect halfway to the margin about 4× that,
  and an effect at 90% of the margin about 100×.
- **Transferability.** The offline figures are modeled probabilities for a single carousel with all 12 cards seen.
  The A/B metric is an observed stream from the first real carousel within 30 minutes, where viewing is partial and
  behavior is real. The offline changes are planning scenarios, not forecasts of the A/B effect.
- **Verification.** `ni_sample_size_per_arm` in `src/metrics.py` is unit-tested against a hand calculation. A
  simulation test confirms it delivers the stated power and about 2.5% type I error at the margin.
- Ratio secondary metrics (e.g. streams per placement) need the delta method. CUPED with pre-period streaming can
  reduce n.
- The plan logs viewed positions per render as a diagnostic (never a filter), because the offline result depends
  on whether listeners scroll past position 3.

## 11. Originality

The engagement model and data are Deezer's and are cited. The exposure-aware ranker, the loss-budget presets, the
metrics suite, the segment analysis, the robustness checks, the tradeoff curve and the dashboard are independent
work. They do not reproduce the paper's bandit comparisons or figures.
