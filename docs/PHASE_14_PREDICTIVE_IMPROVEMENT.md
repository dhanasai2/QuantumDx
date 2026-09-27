# Phase 14 (Predictive-Improvement Track): Systematic Search for a Defensible Improvement

**Status:** Complete — **Decision: B — NO DEFENSIBLE IMPROVEMENT FOUND**
**Scope:** A systematic, leakage-safe investigation of whether any technically defensible modeling change (backbone, features, fusion) improves on the Phase 11 XGBoost baseline. Distinct from the product-build "Phase 14" track (`src.large_dataset.phase14_hybrid_product`); by explicit instruction this experimentation runs first, and the product build will be adapted afterward to whatever architecture this evidence supports.
**Code:** [`src/large_dataset/phase14_predictive_experiments.py`](../src/large_dataset/phase14_predictive_experiments.py)
**Tests:** [`tests/test_phase14_predictive_experiments.py`](../tests/test_phase14_predictive_experiments.py) (12)
**Results:** `results/large_dataset/phase14_predictive_experiments/`

---

## 1. Protocol

All model/feature selection uses 5-fold stratified CV on the full training pool (n=66,641) only. The fixed 200-row test set (fingerprint `96eac11a8394b87e`) is touched **only if** CV evidence shows a consistent, seed-stable improvement — enforced in code, not just by discipline: `test_evaluation` and `statistics` are `None` in the persisted summary whenever `consistent_improvement_in_cv` is `False`.

## 2. Stage 1 — Diagnostic error analysis (read-only)

Analyzed Phase 11's already-persisted XGBoost test predictions (no new training). Findings:
- **Confidence by outcome**: FPs are confidently wrong (mean proba 0.722), FNs moderately wrong (mean 0.341) — not simply borderline uncertainty.
- **Calibration**: reasonably good already — no large systematic bias across 8 quantile bins.
- **Threshold behavior**: errors concentrate near the decision boundary (38% error rate for |p−0.5|<0.1) far more than away from it (11% error rate) — consistent with genuine class overlap, not miscalibration.
- **Feature profile by outcome**: FN patients' mean BP/cholesterol profile resembles the TN (healthy) group; FP patients' profile resembles the TP (disease) group. **Most residual errors are patients whose measured risk factors don't match their actual outcome** — consistent with irreducible noise from unmeasured factors (genetics, family history, diagnostic variability), not a fixable representation gap.

This diagnosis predicted, before any new model was trained, that large gains were unlikely — a prediction the subsequent stages confirmed.

## 3. Stage 2 — Backbone comparison (5-fold CV)

| Candidate | CV ROC-AUC | CV PR-AUC |
|---|---:|---:|
| XGBoost (Phase 11 config) | 0.8015 ± 0.0063 | 0.7836 ± 0.0061 |
| HistGradientBoostingClassifier (sklearn, no new dependency) | 0.8009 ± 0.0060 | 0.7834 ± 0.0057 |

Difference (0.0006) is far smaller than the fold-to-fold standard deviation (~0.006) — not a real difference.

## 4. Stage 3 — Feature engineering (5-fold CV)

Added standard, clinically-established cardiovascular indicators (BMI, pulse pressure, mean arterial pressure, age×cholesterol interaction) — not invented to chase AUC.

| Configuration | CV ROC-AUC |
|---|---:|
| Without engineered features | 0.8015 ± 0.0063 |
| With engineered features | 0.8015 ± 0.0064 |

Identical to the 4th decimal place. XGBoost's own tree splits already capture whatever nonlinear signal these ratios/interactions would add — unsurprising for a gradient-boosted tree model, which can already approximate such interactions from the raw inputs.

## 5. Stage 4 — Fusion redesign (5-fold CV)

Compared Phase 13's Logistic Regression fusion against a small XGBoost fusion, both on the identical OOF-derived `[p_xgb, refinement_proxy, margin]` input (isolating the fusion *architecture* as the only variable; the refinement term was fixed at zero since Phase 13 already established it contributes negligibly).

| Fusion head | CV ROC-AUC |
|---|---:|
| Logistic Regression (Phase 13's choice) | 0.8015 ± 0.0063 |
| XGBoost | 0.8012 ± 0.0064 |

No improvement from a more expressive fusion model either.

## 6. Stage 5 — Multi-seed CV stability check

Since no candidate beat the baseline in Stage 2–4, the "candidate" and "baseline" in this check are the same model (Phase 11's XGBoost) — confirmed extremely stable across 5 seeds (42, 123, 2024, 7, 99): **mean CV ROC-AUC 0.80147, std 0.00017** across seeds. This stability is itself useful evidence: the baseline's CV performance is not seed-sensitive, so the "no improvement" finding in Stages 2–4 is not an artifact of unlucky CV splits.

## 7. Stage 6 — Test-set evaluation

**Not performed.** Per the explicit protocol, the test set is only touched when CV evidence supports a candidate. None did, so `test_evaluation` and `statistics` are `null` in the persisted summary — the test set remains completely untouched by this phase.

## 8. Runtime

Total: ~182 seconds (~3 minutes) — Stage 2 (17.8s), Stage 3 (25.7s), Stage 4 (14.9s), Stage 5 (124.0s, the 5-seed×5-fold sweep). Every candidate model (XGBoost, HistGradientBoosting) fits in 3–4 seconds at full scale — this was one of the cheapest phases in the project, a direct consequence of not touching the expensive quantum machinery at all.

## 9. Final decision

**B — NO DEFENSIBLE IMPROVEMENT FOUND.**

None of the three investigated levers (classical backbone choice, clinically-motivated feature engineering, fusion-head architecture) produced a CV improvement exceeding the measurement noise, despite each being individually well-motivated and cheaply testable. Combined with Phase 9–13's already-exhaustive finding that no tested quantum component adds value either, the diagnostic in Stage 1 is the most likely explanation: the residual gap between this dataset's available 11 features and perfect prediction is substantially explained by patients whose measured risk factors do not match their actual disease status — a pattern consistent with irreducible noise from factors this dataset does not capture (family history, genetics, diagnostic variability), not a fixable modeling deficiency.

**Recommendation**: Phase 11's XGBoost (full classical features, ROC-AUC 0.8567 on the fixed test set) remains the strongest, most defensible model. Further tuning along these three axes is unlikely to yield a meaningful improvement without a fundamentally different data source (e.g., additional clinical variables not in this dataset) — not a modeling change this phase could make. This supports building the product (the next step, per instruction) around the already-validated Phase 11 XGBoost backbone, optionally still presenting the hybrid architecture as a genuine engineering/demonstration component (per Phase 13's own conclusion) but without claiming it improves accuracy.

## Files created

```
src/large_dataset/phase14_predictive_experiments.py
tests/test_phase14_predictive_experiments.py
docs/PHASE_14_PREDICTIVE_IMPROVEMENT.md                (this file)
results/large_dataset/phase14_predictive_experiments/
    diagnostic_analysis.json, cv_comparison_table.csv, phase14_experiments_summary.json
```

No Phase 9, 11, or 13 artifact was modified or the test set touched — verified by `tests/test_phase14_predictive_experiments.py::test_diagnostic_analysis_does_not_modify_phase11_file_if_present` and `test_test_set_only_touched_when_cv_supports_a_candidate_if_present`.
