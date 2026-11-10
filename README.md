# Beyond the Algorithm

**The engagement–discovery tradeoff in music streaming**

*Can a music platform spread recommendations across much more of its catalog without meaningfully sacrificing
predicted listener engagement, and who pays for it if it does?*

An independent product analytics case study built on real, anonymized Deezer user and playlist features
(974,960 users, 862 playlists). It compares three carousel recommendation strategies on the same listeners and the
same 12 slots, maps the full engagement-vs-discovery tradeoff curve, and turns the result into a product
recommendation and an A/B test plan.

## Why I built this

As a DJ and music enthusiast, I'm interested in how people discover music and how recommendation systems shape
what gets heard. I wanted to explore the tension between recommending what listeners are already likely to enjoy
and creating opportunities for a broader range of music to reach them.

Using Deezer's public research data, I investigated how changing recommendation priorities affects predicted
engagement, catalog representation, and the distribution of those tradeoffs across listeners.

---

> **Important:** engagement figures are **model-based expected values** from Deezer's published simulator,
> applied to real anonymized features. They are not observed streams and not results of Deezer's live experiments.
> This project is not affiliated with Deezer.

---

## Key findings

| Strategy | λ | Predicted engagement | Change | Catalog placed (12 slots) | Catalog on first screen (3 slots) | Effective variety |
|---|---|---|---|---|---|---|
| Engagement-first | 0 | 74.3% | — | 11% | 5.5% | 26 |
| **Balanced discovery** | 0.3 | 73.2% | **−1.5% (−1.1 pts)** | **23%** | 6.8% | **45** |
| Diversity-first | 1.0 | 65.0% | −12.5% (−9.3 pts) | 65% | 25.8% | 181 |

*Mean of 5 disjoint samples of 50,000 users. Predicted engagement = modeled P(stream ≥ 1 of the 12 carousel cards).
A "placement" is a playlist assigned to a slot, not a confirmed view.*

1. **Placements are highly concentrated.** A pure engagement ranker places only ~11% of the catalog for anyone.
   Its placements are as concentrated as if only 26 playlists were shown evenly.
2. **Placement variety is cheap at first.** Balanced discovery doubles the catalog placed and lifts effective
   variety 77% for a 1.5% drop in predicted engagement.
3. **But most of it lands beyond the first screen.** The catalog in the first 3 cards moves only from 5.5% to 6.8%,
   and first-screen engagement changes −0.12%. These are recommendation placements, not observed views or
   discovery. The data has no impression or scroll records, so actual viewed exposure is unknown and would need
   to be measured online.
4. **The cost is regressive.** At the balanced setting, the least-likely-to-stream quartile loses 7.7% (−3.2 pts).
   The most-likely quartile loses 0.02%.
5. **Recommendation:** test balanced discovery online with a 2% non-inferiority margin on first-carousel
   streaming, log viewed positions, and guard the bottom quartile. Don't make Diversity-first the home-screen default.

## The dashboard

| Tab | What it shows |
|---|---|
| **Summary** | The question, three headline numbers, a 862-square catalog map (all 12 cards vs. first 3), the proposed decision, why discovery matters |
| **Explore strategies** | Pick a strategy or custom λ and a listener group; KPIs, placement concentration (12 vs. 3 slots), who pays (relative % or pts) |
| **The tradeoff** | Interactive tradeoff curve with a loss-budget slider and three placement-variety measures; quartile curves; 100-cluster view |
| **Recommendation** | Findings, actions, measured-vs-hypothesis table, assumptions and limitations |
| **A/B test plan** | Non-inferiority design, guardrails, validation checks, decision rules, consistent sample-size calculator |
| **Methodology** | Data and model provenance, our ranker, robustness checks, definitions, reproducibility |

## Approach

**Engagement model (Deezer's, used as published).** p(u, i) = σ(xᵤ · θᵢ), where xᵤ is a 96-dim user embedding plus
a bias term and θᵢ a 97-dim playlist weight vector. Slate reward = 1 − ∏(1 − pᵢ).

**Exposure-aware ranker (this project's own).** Users are served sequentially. Each playlist is discounted by how
over-exposed it already is relative to an even share:

```
score(u, i) = log p(u, i) − λ · log(1 + placements_i / even_share)
```

λ = 0 is engagement-first. Larger λ trades modeled engagement for a flatter placement distribution. The presets are
picked by rule: *Balanced* = the most diverse λ within a 2% loss budget, *Diversity-first* = within 15%.

**Diversity metrics.** Catalog placed (coverage), first-screen catalog, effective variety (exp of placement entropy),
Gini and top-10% share. Every strategy makes exactly users × 12 placements, so the metrics measure *how placements
are distributed*, not how many cards are shown. Exposure is reported over all 12 slots and over the 3 slots visible
without scrolling, because the data has no scroll behavior.

**Robustness.** Results are stable across 5 user samples and 5 serving orders. At high λ, users served early bear
more of the cost (see methodology §8).

Full details are in [`docs/methodology.md`](docs/methodology.md).

## Project structure

```
beyond-the-algorithm/
├── app.py                    # Streamlit dashboard (reads data/processed only)
├── src/
│   ├── metrics.py            # engagement + exposure/diversity metrics
│   ├── strategies.py         # engagement-first, exposure-aware ranker, random reference
│   └── analysis.py           # runs all strategies on one user sample; segments; presets
├── scripts/prepare_data.py   # offline pipeline: raw CSVs -> data/processed (~1 min)
├── data/
│   ├── raw/                  # Deezer CSVs (git-ignored, 1.8 GB)
│   └── processed/            # small result tables committed for deployment (<1 MB)
├── tests/                    # pytest: metrics + strategy behavior
├── docs/methodology.md
└── .streamlit/config.toml    # dark theme
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: rebuild results from raw data (requires the Zenodo files in data/raw/)
python scripts/prepare_data.py

pytest -q
streamlit run app.py
```

The processed tables are committed, so the app runs without the raw data.

**Getting the raw data:** download `user_features.csv` and `playlist_features.csv` from
[Zenodo 10.5281/zenodo.4048678](https://zenodo.org/records/4048678) into `data/raw/`.

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub. `data/raw/` is git-ignored. Make sure `data/processed/` **is** committed.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public app from GitHub**.
3. Select the repo and branch, set **Main file path** to `app.py`, and under *Advanced settings* choose Python 3.12
   (the pinned NumPy 2.0.2 has no wheels for 3.13).
4. Deploy. The app reads only the <1 MB of processed CSVs, so it starts in seconds. No secrets are needed.

## Limitations

- Outcomes are modeled, not observed. The model is least reliable for rarely shown playlists, which are exactly
  the ones diversification promotes.
- The analysis is static and single-session: no feedback loops, novelty, fatigue or retention.
- Coverage is measured per 50k-user sample and grows with population size. Compare strategies with each other.
- Playlists and clusters are anonymized. Coverage is **not** genre diversity or emerging-artist exposure.
- The variation across samples reflects sampling stability of a deterministic calculation. It is not a
  confidence interval for a real treatment effect.

## Data and attribution

Data: Deezer, *Datasets from the RecSys 2020 article "Carousel Personalization in Music Streaming Apps with
Contextual Bandits"*, Zenodo, [10.5281/zenodo.4048678](https://zenodo.org/records/4048678), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw data is not redistributed in this repo.

```bibtex
@inproceedings{bendada2020carousel,
  title     = {Carousel Personalization in Music Streaming Apps with Contextual Bandits},
  author    = {Bendada, Walid and Salha, Guillaume and Bontempelli, Theo},
  booktitle = {14th ACM Conference on Recommender Systems (RecSys 2020)},
  year      = {2020}
}
```

The engagement model follows the authors' published simulator
([deezer/carousel_bandits](https://github.com/deezer/carousel_bandits)). The strategies, metrics, analysis and
dashboard are independent work.
