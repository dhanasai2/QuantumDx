# Stage D: Controlled Scaling Experiment at n=20,000

**Status:** Complete
**Scope:** The fourth measured point (n_train=20,000) in the QSVM-vs-classical scaling experiment, on the IDENTICAL 200-row held-out test set established at Stages A/B/C (fingerprint `96eac11a8394b87e`). Does not change the dataset, the fixed test sets, the quantum methodology (4 qubits, `zz_feature_map`, reps=2, linear entanglement, PCA-4, C-grid), or the Phase 3-5 classical methodology, except for one documented, narrowly-scoped computational-budget deviation (below).
**Code:** [`src/large_dataset/stage_d.py`](../src/large_dataset/stage_d.py)
**Tests:** [`tests/test_stage_d.py`](../tests/test_stage_d.py) (18 tests)
**Results:** `results/large_dataset/stage_d/`, `results/large_dataset/statistical_robustness/stage_d_extension/`

---

## 1. Objective

Stages A, B, and C showed the classical-vs-QSVM ROC-AUC gap narrowing (0.1062 → 0.0788 → 0.0503), a 52.6% reduction from A to C — but three points cannot establish a trend. Stage D's sole question: **at n_train=20,000, does the gap continue narrowing, plateau, or widen?** All three outcomes were treated as equally acceptable before the experiment ran.

## 2. Dataset and split verification

- Training pool: 66,641 rows (unchanged).
- Stage D training subset: exactly 20,000 rows, stratified, seed 42, drawn via the same `nested_stratified_stage_samples` function Stages A/B/C used (no new sampling code was needed).
- **Nesting verified programmatically**: Stage A (1,000) ⊂ Stage B (5,000) ⊂ Stage C (10,000) ⊂ Stage D (20,000) — `nesting_verified: {"1000": true, "5000": true, "10000": true, "20000": true}`.
- Stage D's training set has **zero overlap** with both fixed test sets (checked before any model touched the data; the run raises `ValueError` otherwise).
- Comparison-set fingerprint: `96eac11a8394b87e` (matches expected), n=200, positive rate=0.495 — identical to every prior stage, confirmed via SHA-256 of sorted row ids.

## 3. Computational budget deviation (Section 8) — the only methodology change

**What changed:** RBF-SVM's hyperparameter grid was narrowed from the established 4×4=16 combinations (`C ∈ {0.1,1,10,100}`, `gamma ∈ {scale,0.01,0.1,1}`) to the single combination `C=10.0, gamma=0.01` — the value Stage C's own full, unmodified grid search already selected as best (`results/large_dataset/corrected_comparison/stage_10000/corrected_comparison.json → best_params.rbf_svm`). Logistic Regression, Random Forest, and XGBoost retain their full, unreduced grids.

**Why it was necessary (measured, not assumed):** Before committing to the full experiment, a single representative fit (`C=10, gamma='scale'`, one real 16,000-row fold — matching GridSearchCV's actual internal fold size at n=20,000) was benchmarked in isolation: **68.8 seconds** with scikit-learn's default kernel cache, **126.3 seconds** with a 15× larger cache (ruling out cache starvation as a lever — a bigger cache made it slower here). Extrapolated across the full grid (16 combinations × 5 outer folds = 80 fits, each *also* performing its own internal 5-fold Platt probability calibration since `probability=True`), the full search implied a multi-hour-to-multi-day budget. A first, unmodified attempt at the full grid was run and **stopped after 4 hours 8 minutes of continuous CPU-bound execution with no completion in sight** — confirming infeasibility empirically rather than by extrapolation alone.

**Verification before use on real data:** a small-n (n=300) dry run of the modified code confirmed (a) `rbf_svm`'s `best_params` come back as exactly the frozen `{C: 10.0, gamma: 0.01}`; (b) every other model and the QSVM path are byte-identical to pre-fix behavior (QSVM ROC-AUC reproduced exactly: 0.6502150215021502); (c) the deviation is recorded in `stage_d_summary.json`.

**Explicit caveat:** RBF-SVM's Stage D result is under a **different search budget** than Stage A/B/C (1 combination vs. 16) — its Stage D wall time (532.5s) must not be compared to its Stage A/B/C wall time as if the budgets were equal, and its result should be read as "the previously-best hyperparameters, retrained on more data," not as a fresh search at n=20,000.

## 4. Exact Stage D methodology

Everything else reused Stages A/B/C's own code, unmodified: `SharedFeaturePipeline`/`build_quantum_pipeline` (Phase 2), `run_grid_search` (Phase 3, 5-fold CV, preprocessing refit per fold, LR/RF/XGB's full original grids), the fidelity kernel and feature map (Phase 4/5), `run_qsvm_grid_search`, and the paired-statistics functions from `corrected_comparison.py` / `statistical_robustness.py`. One genuinely new component: `kernel_matrix_from_statevectors_blockwise` (float32, block_size=2000) replaces the whole-matrix vectorized kernel used at Stages A/B/C — required because the whole-matrix approach needs a transient complex-valued (20000,20000) intermediate plus multiple float64 copies (10+ GB) on a machine with far less headroom available; verified numerically equivalent to the float64 reference in `tests/test_quantum.py` before use.

## 5. Stage D classical results (identical 200-row test set)

| Model | ROC-AUC | 95% CI | PR-AUC | Sens | Spec | Acc | Best params |
|---|---:|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.8500 | [0.7959, 0.8986] | 0.8501 | 0.7273 | 0.8020 | 0.765 | C=1.0 |
| RBF-SVM | 0.8406 | [0.7830, 0.8934] | 0.8381 | 0.7273 | 0.8218 | 0.775 | C=10.0, gamma=0.01 (frozen, §3) |
| Random Forest | 0.8536 | [0.7990, 0.9033] | 0.8500 | 0.7172 | 0.8614 | 0.79 | max_depth=5, n_estimators=300, max_features=sqrt |
| **XGBoost (best)** | **0.8613** | [0.8064, 0.9109] | 0.8422 | 0.7576 | 0.8317 | 0.795 | max_depth=4, lr=0.05, n_estimators=100 |

## 6. Stage D QSVM results (identical 200-row test set)

| Metric | Value | 95% CI |
|---|---:|---|
| ROC-AUC | 0.7995 | [0.7327, 0.8603] |
| PR-AUC | 0.8000 | [0.7135, 0.8721] |
| Sensitivity | 0.7576 | [0.6730, 0.8400] |
| Specificity | 0.7624 | [0.6778, 0.8404] |
| Accuracy | 0.76 | [0.70, 0.8151] |

Best QSVM `C=1.0` (selected by CV over the unmodified 5-value C-grid — no reduction applied to QSVM's own search).

## 7. C→D and A→D gap changes

| Stage | n_train | QSVM ROC-AUC | Best classical | Best ROC-AUC | Gap |
|---|---:|---:|---|---:|---:|
| A | 1,000 | 0.7391 | Logistic Regression | 0.8453 | 0.1062 |
| B | 5,000 | 0.7715 | Logistic Regression | 0.8503 | 0.0788 |
| C | 10,000 | 0.8013 | XGBoost | 0.8516 | 0.0503 |
| D | 20,000 | 0.7995 | XGBoost | 0.8613 | **0.0618** |

Descriptive gap-change arithmetic (not a fitted trend):

| Transition | Gap change | % change |
|---|---:|---:|
| A→B | −0.0274 | −25.80% (narrowed) |
| B→C | −0.0285 | −36.17% (narrowed) |
| **C→D** | **+0.0115** | **+22.86% (widened)** |
| A→D | −0.0444 | −41.81% (net narrowing, non-monotonic) |

**QSVM's ROC-AUC was essentially flat from C to D** (0.8013 → 0.7995, Δ=−0.0018, well within both stages' overlapping confidence intervals), while **best-classical (XGBoost) continued improving** (0.8516 → 0.8613, Δ=+0.0097). The gap widened because classical kept improving while QSVM plateaued, not because QSVM got worse in any dramatic sense.

No formal paired test was run on the *change in gap* between stages C and D specifically (that would require a different hypothesis test than "is classical > QSVM at this one stage"); the C→D reversal is reported descriptively, and should be read as one additional observed point, not a proven trend break.

## 8. Statistical significance results (Stage D, vs. QSVM, on 200-row set)

| Classical model | DeLong Δ | DeLong p | Bootstrap Δ | Bootstrap p | McNemar p (n discordant) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0505 | 0.0086 | 0.0505 | 0.008 | 1.000 (21) |
| RBF-SVM | 0.0411 | 0.0215 | 0.0411 | 0.014 | 0.629 (17) |
| Random Forest | 0.0541 | 0.0023 | 0.0541 | 0.002 | 0.263 (20) |
| XGBoost | 0.0618 | 0.0009 | 0.0618 | 0.001 | 0.143 (17) |

All four classical models beat QSVM significantly on raw (uncorrected) DeLong and bootstrap p-values. McNemar is not significant for any of the four at Stage D — consistent with the same DeLong/bootstrap-vs-McNemar disagreement pattern established and explained in Phase 6A (§10 of `STATISTICAL_ROBUSTNESS.md`): decision-level agreement is high while a real ranking-level gap persists.

## 9. Holm-corrected results — the 16-comparison family

Per Task 15, Phase 6A's original 12 comparisons (Stages A/B/C, read-only, never modified) were combined with Stage D's 4 new comparisons into **one 16-comparison family**, and Holm-Bonferroni was re-applied once across all 16 — not to Stage D alone.

| Stage | Model | DeLong p (Holm) | Sig. | Bootstrap p (Holm) | Sig. |
|---|---|---:|:---:|---:|:---:|
| A | all 4 | ≤0.0051 | Yes | ≤0.0051 | Yes |
| B | all 4 | ≤0.0086 | Yes | ≤0.0090 | Yes |
| **C** | **all 4** | **0.0514** | **No** | **0.06** | **No** |
| D | Logistic Regression | 0.0514 | No | 0.048 | **Yes** |
| D | RBF-SVM | 0.0514 | No | 0.06 | No |
| D | Random Forest | 0.0162 | **Yes** | 0.014 | **Yes** |
| D | XGBoost | 0.0082 | **Yes** | 0.009 | **Yes** |

**Result: 10/16 significant by DeLong, 11/16 by bootstrap** (down from 12/12 when only Phase 6A's 12 comparisons were the family). This is a genuine, reportable shift, not an error: enlarging the family from 12 to 16 hypotheses makes the Holm correction strictly more conservative (the smallest p-value now needs to clear a 1/16 threshold step instead of 1/12), and **all four of Stage C's comparisons — previously significant — no longer survive correction** under the larger family, while **Stage D's Random Forest and XGBoost comparisons do survive** (and Logistic Regression survives under bootstrap only). This illustrates a real, non-obvious consequence of pre-committing to a growing family of hypotheses across an ongoing experiment: statistical conclusions about earlier stages can retroactively change as later stages are added to the same correction family. Full detail: `results/large_dataset/statistical_robustness/stage_d_extension/auc_comparisons_16.csv`.

## 10. Confidence intervals

Bootstrap 95% CIs (2,000 resamples, seed 42, identical methodology to Phase 6A) for all 7 metrics × all 5 models at Stage D are in §5/§6 (ROC-AUC/PR-AUC/Sens/Spec/Acc) and in `stage_d_summary.json → statistics.single_model_cis` (adds precision and F1). All CIs are wide (n=200) — QSVM's and XGBoost's ROC-AUC CIs at Stage D ([0.7327, 0.8603] and [0.8064, 0.9109]) barely overlap, consistent with — but not dramatically more powerful than — the DeLong/bootstrap point-estimate comparison in §8.

## 11. Confusion-matrix interpretation

At the fixed 0.5 threshold: QSVM produces 24 false positives and 24 false negatives (balanced error), while XGBoost produces 17 false positives and 24 false negatives (fewer false positives, same false negatives) — XGBoost's specificity advantage (0.832 vs. 0.762) is the larger contributor to its ROC-AUC lead over QSVM's comparable sensitivity (0.758 vs. 0.758, effectively tied). See `results/large_dataset/stage_d/confusion_matrices.png`.

## 12. ROC/PR interpretation

QSVM's ROC and PR curves at Stage D sit visibly below all four classical models' curves across most of the operating range, consistent with the AUC gap — see `results/large_dataset/stage_d/roc_curves.png` and `pr_curves.png`. No crossing pattern that would suggest QSVM outperforms classical models in a specific operating region.

## 13. Kernel diagnostics

| Property | Value |
|---|---|
| Shape | 20,000 × 20,000 |
| dtype | float32 |
| Memory | 1,600 MB (exactly 20000²×4 bytes) |
| Symmetric | Yes |
| Diagonal mean / max deviation from 1.0 | 1.0 / 0.0 |
| Min / max / mean value | 4.42e-11 / 1.0 / 0.1308 |
| Off-diagonal mean / std | 0.1308 / 0.1613 |
| Within valid fidelity range [0,1] | Yes |

The blockwise, float32 kernel assembly performed exactly as designed: train-train kernel assembly took **9.2 seconds** (vs. an estimated 10+ GB / infeasible for the whole-matrix float64 approach at this size), and process RSS at that checkpoint was only 1,857 MB.

## 14. Runtime and memory

| Phase | Time |
|---|---:|
| Data loading + split | 0.3s |
| Classical: Logistic Regression | 38.1s |
| Classical: RBF-SVM (narrowed grid, §3) | 532.5s |
| Classical: Random Forest | 472.2s |
| Classical: XGBoost | 331.6s |
| **Classical phase total** | **1,459.3s (24.3 min)** |
| QSVM: preprocessing + PCA | 1.2s |
| QSVM: statevectors (train + test) | 37.6s |
| QSVM: kernel assembly (train-train + test-train) | 9.3s |
| QSVM: CV C-grid search | 1,734.0s |
| QSVM: final inference | 0.09s |
| **QSVM phase total** | **1,806.1s (30.1 min)** |
| **Stage D total** | **~54.4 min** |

Runtime must be read under each branch's own search budget (§3 caveat) — the QSVM phase's dominant cost is CV tuning (25 fits over a 5-value C-grid on the precomputed kernel), not kernel assembly, which remained cheap throughout.

**Memory — an observed near-miss, reported honestly.** During the QSVM CV phase, OS-level process memory (Windows `Get-Process`, external to the script) reached **~6.2 GB private memory**, and system-wide free memory dropped to **as low as 0.84 GB out of 16 GB total** — flagged in real time by a peer session's independent monitoring, under the hypothesis that `SVC(kernel="precomputed")`'s final full-matrix refit upcasts the float32 kernel to float64 internally, transiently holding both a 1.6 GB and a 3.2 GB copy. The run completed successfully without an OOM crash, and the script's own internal `psutil` RSS checkpoints (254 MB → 1,857 MB after kernel assembly → **1,666 MB after CV tuning**, i.e. no sustained increase) do not show the predicted spike — meaning either the spike was transient and reclaimed between checkpoints, or the mechanism differs from the hypothesis. **This discrepancy between the internal (`psutil`) and external (OS-level) memory views is unresolved and flagged as a fragility**: this run had no margin to spare, and a future, larger-scale run on this same machine should not assume the same near-miss will resolve safely again without addressing it directly (e.g., restructuring the fit call to release the float32 kernel reference before `search.fit()`, or running with more available system memory).

## 15. A/B/C/D comparison table

| Stage | n | QSVM ROC-AUC | Best classical | Best ROC-AUC | Gap | QSVM Sens | QSVM Spec |
|---|---:|---:|---|---:|---:|---:|---:|
| A | 1,000 | 0.7391 | Logistic Regression | 0.8453 | 0.1062 | 0.7172 | 0.6832 |
| B | 5,000 | 0.7715 | Logistic Regression | 0.8503 | 0.0788 | 0.7374 | 0.7426 |
| C | 10,000 | 0.8013 | XGBoost | 0.8516 | 0.0503 | 0.7475 | 0.7525 |
| D | 20,000 | 0.7995 | XGBoost | 0.8613 | 0.0618 | 0.7576 | 0.7624 |

These are four observed measurements across training sizes, not a fitted scaling law — no regression line is fit to them, and none should be extrapolated beyond n=20,000.

## 16. Scientific interpretation

1. **The A→C gap-narrowing trend did not continue at Stage D.** It reversed: the gap widened from 0.0503 to 0.0618.
2. **QSVM's own ROC-AUC plateaued** (0.8013→0.7995, a change smaller than either stage's confidence interval width) while **classical continued to improve** (best classical 0.8516→0.8613).
3. **Classical models remain statistically ahead of QSVM at Stage D** on raw significance (all 4, p<0.03); under the more conservative, honestly-enlarged 16-comparison Holm family, Random Forest and XGBoost remain significant, Logistic Regression only under bootstrap, and RBF-SVM no longer reaches significance by either method.
4. **The ordering among classical models shifted slightly**: Random Forest overtook Logistic Regression for 2nd place at Stage D (was 3rd at Stage C).
5. Given one reversal at the very last measured point, **this dataset and feature representation do not currently show evidence of a continuing, reliable QSVM improvement trend** with more training data — the honest reading is "improved from A to C, then plateaued/reversed at D," not "on track to close the gap."

## 17. Limitations

- Four points, one of them (D) breaking the prior pattern, is not enough to distinguish "genuine plateau," "noise," or "temporary dip before further improvement" — only a Stage E (or repeated Stage D with a different seed) could disambiguate, and none is planned or authorized here.
- n=200 bounds precision at every stage; Stage D's own QSVM and XGBoost 95% CIs are wide and only barely separate.
- RBF-SVM's Stage D result used a narrowed 1-combination search (§3) — it is not evidence about what a full search at n=20,000 would have selected or achieved, only about how the previously-best hyperparameters perform on more data.
- The Holm-family-size sensitivity documented in §9 is itself a limitation of the overall multi-stage design: whether a comparison is called "significant" depends on how many other comparisons are bundled into the same correction family, a property of the analysis plan as much as of the data.
- The memory near-miss in §14 is unresolved; this run succeeded, but margin was thin, and the discrepancy between internal and external memory readings was not root-caused.
- Multiple Claude sessions were briefly and inadvertently running duplicate/colliding attempts at this same experiment before a single owner was established (documented for provenance/reproducibility transparency); the final, reported run was executed by a single, coordinated process with no concurrent writers to its output paths.

## 18. Reproducibility

- Seed: 42 throughout (split, sampling, CV, bootstrap).
- Comparison-set fingerprint: `96eac11a8394b87e` (n=200, positive rate 0.495) — verified identical to Stages A/B/C before scoring.
- Nesting verified programmatically for all four stage sizes.
- All hyperparameters selected by CV are recorded in `results/large_dataset/stage_d/stage_d_summary.json → best_params`.
- The one non-standard choice (RBF-SVM's narrowed grid) is fully sourced to Stage C's own prior result and recorded in `computational_budget_deviations`.
- Training ids for Stage D are recoverable deterministically via `nested_stratified_stage_samples(..., seed=42)` on the training pool — not separately persisted as a raw id list, consistent with Stages A/B/C's own convention.

## 19. Recommendation

The evidence does not support continuing to scale the fidelity-kernel QSVM further on this dataset/feature representation as the primary path forward: the gap it was closing from A to C stopped closing at D. Per the project's own stated principle (never proceed to VQC/QNN merely because QSVM underperforms, and only after identifying a concrete, falsifiable hypothesis for why a trainable variational circuit would overcome the specific limitation observed here), **this document does not recommend starting VQC/QNN work on the strength of Stage D alone** — that would require its own justification, separate from "QSVM plateaued." The most defensible immediate next step, if further investigation of this direction is wanted, is a repeated measurement at n=20,000 with a different seed (to distinguish genuine plateau from a one-off dip) before any architectural change is considered. No such repeat, nor any VQC/QNN/hardware/API/dashboard work, has been started here, per the explicit Stage D stop condition.

## Files created

```
src/large_dataset/stage_d.py
tests/test_stage_d.py
docs/STAGE_D.md                                              (this file)
results/large_dataset/stage_d/
    predictions.csv, metrics.csv, kernel_diagnostics.json, runtime.json,
    stage_d_summary.json, roc_curves.png, pr_curves.png, confusion_matrices.png,
    _classical_checkpoint.json, _classical_checkpoint_predictions.csv
results/large_dataset/statistical_robustness/stage_d_extension/
    auc_comparisons_16.csv
```

No Phase 6A file (`results/large_dataset/statistical_robustness/auc_comparisons.csv`, 12 rows) or any Stage A/B/C result file was modified — verified both by direct inspection and by `tests/test_stage_d.py::test_extend_holm_family_never_writes_to_phase_6a_source_file`.
