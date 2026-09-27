# Phase 8A: Label-Aware Feature-Map QSVM Screening

**Status:** Complete — outcome: **hypothesis does not survive screening; STOP, do not scale**
**Scope:** A cheap (n_train=2,000) screening test of a new hypothesis: that label-informed feature interactions (interaction information / synergy) make better QSVM entangling structure than the unsupervised MI-adaptive map (Phase 6/7, already found not reproducible at n=20,000). Does not modify Stage A–D or Phase 7 artifacts.
**Code:** [`src/quantum/label_aware_feature_map.py`](../src/quantum/label_aware_feature_map.py), [`src/large_dataset/phase8a_label_aware_screening.py`](../src/large_dataset/phase8a_label_aware_screening.py)
**Tests:** [`tests/test_label_aware_feature_map.py`](../tests/test_label_aware_feature_map.py) (13), [`tests/test_phase8a_label_aware_screening.py`](../tests/test_phase8a_label_aware_screening.py) (7)
**Results:** `results/large_dataset/phase8a_label_aware/`

---

## 1. Hypothesis

"Feature interactions that are informative about the class label may produce a more useful quantum kernel than interactions selected solely from feature-feature mutual information." Tested as a genuinely different criterion from the Phase 6/7 mechanism, not a variant of it: **interaction information (synergy)**, `Synergy(i,j) = I((X_i,X_j);Y) - I(X_i;Y) - I(X_j;Y)`, computed from training data **and training labels** via quantile-discretized, deterministic mutual information (`sklearn.metrics.mutual_info_score`, no random_state needed — unlike the k-NN based unsupervised MI estimator). Validated on a synthetic XOR case before use: a pair that is each marginally near-independent of the label but jointly determines it is correctly identified as top-ranked (`tests/test_label_aware_feature_map.py::test_detects_known_xor_synergy`) — confirming the mechanism measures genuine synergy, not disguised marginal relevance.

## 2. Why n_train=2,000, screening only

Per instruction, this is a cheap first check before any scaling decision. n=2,000 was chosen deliberately distinct from every prior stage (A=1,000, B=5,000, C=10,000, D=20,000). Runtime was estimated before running (from Stage D/Phase 7's measured n=20,000 costs, ~100x smaller kernel and expected much-cheaper CV) and confirmed cheap by an n=200 dry run (~1-2s/model) before committing to the real n=2,000 run.

## 3. Method

One fixed screening training subset (n=2,000, seed 42, drawn via the existing `nested_stratified_stage_samples` on the same training pool, verified zero-overlap with both fixed test sets) with preprocessing/PCA-4 fit once, shared across three feature maps evaluated identically:

1. **Baseline**: `zz_feature_map`, linear entanglement (unchanged).
2. **MI-adaptive** (Phase 6/7 mechanism, unmodified code, **re-derived at n=2,000** for a fair same-training-size comparison — not reused from the n=20,000 result).
3. **Label-aware** (new mechanism, this phase).

All three: 4 qubits, reps=2, identical 3-ZZ-term budget, same memory-safe blockwise kernel, same C-grid/5-fold CV/seed. The 200-row test set (fingerprint `96eac11a8394b87e`) was touched exactly once per model, after CV.

## 4. Selected structures

| Map | Selected pairs |
|---|---|
| Baseline (fixed) | (0,1), (1,2), (2,3) |
| MI-adaptive | (1,2), (0,1), (0,3) |
| Label-aware | (1,2), (0,3), (0,2) |

Marginal label MI (label-aware map): feature 0 = 0.106 (by far the most individually predictive), features 1–3 all < 0.01. The synergy matrix's largest entry is (1,2) at 0.026 — an order of magnitude smaller than feature 0's marginal relevance, meaning no feature pair carries substantially more *joint* label information than feature 0 carries *alone*. This is itself informative: at n=2,000, this particular PCA representation does not contain a strong pairwise synergy signal for the label (unlike the synthetic XOR validation case, which was constructed to have one).

## 5. Results (n_train=2,000, identical 200-row test set)

| Model | ROC-AUC | 95% CI | PR-AUC | Sens | Spec | Acc | F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Baseline | 0.7379 | [0.6614, 0.8106] | 0.6920 | 0.7172 | 0.6931 | 0.705 | 0.7065 |
| MI-adaptive | 0.7666 | [0.6917, 0.8338] | 0.7015 | 0.7374 | 0.7228 | 0.730 | 0.7300 |
| Label-aware | 0.7475 | [0.6757, 0.8151] | 0.7297 | 0.6667 | 0.6931 | 0.680 | 0.6735 |

## 6. Statistical comparison vs. baseline

| Variant | Δ ROC-AUC | DeLong p | Bootstrap p (ROC) | Bootstrap p (PR) | McNemar p | Holm-adjusted DeLong p |
|---|---:|---:|---:|---:|---:|---:|
| MI-adaptive | +0.0287 | 0.235 | 0.250 | 0.837 | 0.424 | 0.471 |
| Label-aware | +0.0096 | 0.721 | 0.683 | 0.430 | 0.487 | 0.721 |

Holm-Bonferroni applied once across both comparisons (one family, α=0.05). **Neither variant reaches significance by any test, before or after correction.** Both bootstrap and DeLong CIs for both variants include zero.

## 7. Runtime and memory

Total: ~113s for all three models (baseline 31.8s, MI-adaptive 40.3s, label-aware 41.2s) — CV tuning dominates each (~29–35s), kernel assembly negligible (<0.2s), memory trivial (16MB kernel, ~245MB process RSS peak). Confirms the pre-run estimate: roughly two orders of magnitude cheaper than the n=20,000 experiments, as expected for a 10x-smaller training set with a kernel that scales quadratically.

## 8. Test-set discipline (verified)

Every function in `label_aware_feature_map.py` accepts training data and training labels only — no parameter named or resembling "test," "proba," or any performance metric exists anywhere in the module (`tests/test_label_aware_feature_map.py::test_no_function_accepts_test_or_performance_data`, structural signature inspection). `y_train`'s presence is a deliberate, documented design choice (unlike the unsupervised MI map, which accepts no label at all) — verified explicitly by a separate test confirming it IS present, so the "no test dependency" guarantee is not confused with "no label dependency."

## 9. Limitations

- Single screening run, single seed (42) — no multi-seed reproducibility check was performed or requested at this stage (Phase 7 already demonstrated why that matters for the MI-adaptive mechanism; the same caution applies here, and is exactly why scaling is not warranted from this one result alone).
- n=2,000 training and n=200 test both bound precision; the observed deltas (+0.0287, +0.0096) are well within each variant's own bootstrap CI width.
- The label-aware mechanism's synergy scores at n=2,000 were modest relative to a single feature's marginal relevance — this screening subset may not be where a synergy-based mechanism has the best chance to show an effect, but re-deriving it at a size where a real synergy signal happens to be stronger was not run, per the explicit instruction not to scale or tune until something wins.

## 10. Whether the hypothesis survives

**It does not, at this screening stage.** The label-aware map underperforms the MI-adaptive map on the primary endpoint (ROC-AUC Δ +0.0096 vs +0.0287) and neither reaches significance vs. baseline. This is not "label-aware feature maps don't work" — it is "this screening run gives no compelling reason to invest further compute in this specific mechanism before doing so has already been earned by better-than-baseline, statistically supported evidence at this cheap scale." Per the explicit instruction: **since neither variant shows a compelling improvement, STOP here — do not scale to n=5,000/10,000/20,000.**

## Files created

```
src/quantum/label_aware_feature_map.py
src/large_dataset/phase8a_label_aware_screening.py
tests/test_label_aware_feature_map.py
tests/test_phase8a_label_aware_screening.py
docs/PHASE_8A_LABEL_AWARE_SCREENING.md                (this file)
results/large_dataset/phase8a_label_aware/
    predictions.csv, metrics.csv, mi_adaptive_map_config.json,
    label_aware_map_config.json, runtime.json, statistics.json,
    phase8a_summary.json, roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D or Phase 7 artifact was read for modification or overwritten — verified by `tests/test_phase8a_label_aware_screening.py::test_stage_and_phase7_artifacts_untouched_if_present`.
