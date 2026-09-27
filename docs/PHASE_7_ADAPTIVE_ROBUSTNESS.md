# Phase 7: Adaptive QSVM Robustness Validation

**Status:** Complete — outcome: **C (adaptive improvement is not reproducible)**
**Scope:** Determines whether the +0.0198 ROC-AUC improvement observed for the adaptive feature map at seed=42 (`docs/ADAPTIVE_QSVM.md`) is a reproducible effect or seed-to-seed noise, by repeating the baseline-vs-adaptive QSVM comparison across 5 independent training seeds at n_train=20,000, on the SAME fixed 200-row test set.
**Code:** [`src/large_dataset/adaptive_robustness.py`](../src/large_dataset/adaptive_robustness.py)
**Tests:** [`tests/test_adaptive_robustness.py`](../tests/test_adaptive_robustness.py) (11)
**Results:** `results/large_dataset/adaptive_robustness/`

---

## 1. Objective

The original adaptive-map experiment (one seed, n=20,000) showed a directionally positive but statistically inconclusive ROC-AUC/PR-AUC improvement. Phase 7 asks: **does this improvement hold up across independent training draws, or was it a one-off?**

## 2. Method

Five pre-specified seeds (7, 21, 42, 84, 123) — no seed selection or cherry-picking. For each: a fresh stratified n_train=20,000 subset drawn from the same fixed training pool (via `nested_stratified_stage_samples(training_pool, target_column, [20000], seed=<seed>)`, the existing, unmodified sampling primitive), verified to have zero overlap with the fixed 200-row test set; preprocessing/PCA refit train-only; the adaptive entangling structure derived from that seed's training data only (same mutual-information mechanism, unmodified); both the baseline (`zz_feature_map`, linear) and adaptive QSVM trained and scored on the identical 200-row test set. **Seed 42 was not recomputed** — it is bit-identical to the already-persisted Stage D baseline and adaptive-experiment results (same training pool, same seed, same sampling function), reused directly rather than wastefully retrained.

## 3. Per-seed results

| Seed | Baseline ROC-AUC | Adaptive ROC-AUC | Δ ROC-AUC | Δ PR-AUC | DeLong p | Bootstrap p | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.8112 | 0.8154 | +0.0042 | −0.0211 | 0.766 | 0.751 | 0.109 |
| 21 | 0.8020 | 0.8085 | +0.0065 | +0.0009 | 0.645 | 0.647 | 0.688 |
| 42 | 0.7995 | 0.8193 | **+0.0198** | +0.0239 | 0.172 | 0.185 | 1.000 |
| 84 | 0.8056 | 0.8026 | −0.0030 | +0.0291 | 0.841 | 0.842 | 1.000 |
| 123 | 0.8032 | 0.7832 | **−0.0200** | −0.0381 | 0.319 | 0.315 | 0.263 |

**0 of 5 seeds reach significance** on DeLong, paired bootstrap, or McNemar — including seed 42, once viewed alongside the other four rather than in isolation. After Holm-Bonferroni correction across the 5 DeLong p-values (one family): **0/5 significant** (largest adjusted p capped at 1.0, smallest raw p=0.172 does not survive correction at α=0.05 with n=5).

## 4. Cross-seed summary

| Metric | Mean Δ | Median Δ | Std | Min | Max | Adaptive better | Adaptive worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROC-AUC | **+0.0015** | +0.0042 | 0.0146 | −0.0200 | +0.0198 | 3/5 | 2/5 |
| PR-AUC | **−0.0010** | +0.0009 | 0.0287 | −0.0381 | +0.0291 | 3/5 | 2/5 |

**The mean effect across seeds is indistinguishable from zero** (+0.0015 ROC-AUC, essentially the sampling noise floor), with a standard deviation (0.0146) nearly 10× the mean — the seed-42 result (+0.0198) sits within one standard deviation of a zero-centered distribution, not as an outlier confirming a real effect. The sign flips: 3 seeds favor adaptive, 2 favor baseline, roughly consistent with a coin flip around zero.

## 5. Adaptive-map stability

| Pair | Selected in (of 5 seeds) |
|---|---:|
| (0,3) | 5/5 |
| (0,1) | 4/5 |
| (1,2) | 4/5 |
| (0,2) | 1/5 (seed 123 only) |
| (2,3) | 1/5 (seed 123 only) |

The selection mechanism is **structurally stable**: 4 of 5 seeds (7, 21, 42, 84) selected the identical pair set {(0,1), (0,3), (1,2)}; only seed 123 selected a different set. This is an important, non-obvious finding in itself: **even among the four seeds that chose the identical adaptive structure, the resulting performance delta ranged from −0.0030 to +0.0198** — i.e., structural stability of the pair selection did **not** translate into stable performance. This rules out "the pair selection itself is noisy" as the sole explanation for the inconsistent result; the performance effect is unstable even when the structure is not.

## 6. Shared test-set limitation

All 5 seeds evaluate on the **same** fixed 200-row test set (fingerprint `96eac11a8394b87e`). This is a training-subset reproducibility/stability analysis, **not 5 independent test evaluations** — the per-seed deltas are correlated through the shared test observations (the same 200 cases' quirks affect every seed's score in a correlated way). This does not overturn the conclusion (the deltas still range in sign and the mean is near zero), but it does mean the 5 "replicates" have less independent statistical information than 5 fully independent test sets would, and the cross-seed std above should be read with that caveat rather than pooled into a naive combined significance test.

## 7. Classical reference (unchanged)

Stage D's best classical model, XGBoost, ROC-AUC = 0.8613 — not retrained here, out of scope for Phase 7. Every adaptive-map seed result (0.78–0.82) remains well below this reference regardless of the adaptive-vs-baseline outcome.

## 8. Optional power-up experiment

Not pursued in this phase. A genuinely independent, larger held-out set (distinct from the existing A–D 200-row and 2,000-row benchmarks) would be required to get more statistical power than the shared 200-row set allows, but constructing one was out of scope for this reproducibility check and is not needed to answer Phase 7's question — the cross-seed *sign instability* (3 up, 2 down, mean ≈ 0) is already a sufficient basis for the conclusion below, independent of any single seed's p-value precision.

## 9. Conclusion

**Outcome C: the adaptive-map improvement is not reproducible.** The original seed=42 result (+0.0198 ROC-AUC) was a plausible-looking but statistically unsupported single observation; across 5 seeds, the mean effect is ≈0, the sign is inconsistent, no seed reaches significance individually or after correction, and — most tellingly — even seeds that produced the *identical* adaptive structure did not produce a stable performance effect. This is not "adaptive feature maps are worse" (Outcome D would require a systematic negative effect, which is not observed either) — it is a genuine null result: on this dataset, PCA representation, and mutual-information selection rule, entangling-structure adaptation does not reproducibly change QSVM performance beyond ordinary training-sample variation.

## 10. Recommendation

Do not pursue further tuning or scaling of this specific adaptive-kernel mechanism — the evidence does not support it as a real effect to refine. Do not proceed to VQC/QNN on the strength of this result either: a null result for one adaptive-kernel mechanism is not evidence for or against a fundamentally different (trainable, variational) quantum model family, and no new hypothesis for why a VQC would behave differently has been established by this experiment. The quantum-kernel-adaptation direction, as tested here, is not scientifically justified to continue without a materially different mechanism or a larger independent test set providing more statistical power.

**STOP per instruction.** No VQC, QNN, hardware, deployment, API, dashboard, or Stage E work has been started.

## Files created

```
src/large_dataset/adaptive_robustness.py
tests/test_adaptive_robustness.py
docs/PHASE_7_ADAPTIVE_ROBUSTNESS.md                (this file)
results/large_dataset/adaptive_robustness/
    per_seed_results.csv, delong_results.csv, bootstrap_results.csv,
    mcnemar_results.csv, adaptive_pair_selection.csv, cross_seed_summary.csv,
    statistical_summary.json, cross_seed_comparison.png,
    delta_roc_auc_across_seeds.png
    seed_7/, seed_21/, seed_42/, seed_84/, seed_123/   (per-seed detail)
```

No Stage A–D or prior adaptive-experiment artifact was modified; seed 42's baseline/adaptive results were read from `results/large_dataset/stage_d/predictions.csv` and `results/large_dataset/adaptive_qsvm/predictions.csv` respectively, never rewritten.
