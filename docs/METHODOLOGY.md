# Methodology and interpretation

## Business objective

Rank sessions by conversion propensity and investigate customer behavior to support prioritization hypotheses. An intervention requires a defined action, cost, capacity and causal experiment. Propensity estimates do not identify who will respond to a discount or message. The historical UCI sample is not a current B2B marketplace sample.

## Data quality and partitions

The downloader retains original bytes and a SHA-256. The schema requires all expected features, positive integer device/source identifiers, valid booleans/categories, finite nonnegative activity and bounded rates. EDA reports missing values, duplicates, dtypes and per-column IQR flags. Raw records remain in descriptive KPIs.

For training, identical **approved predictor vectors** are collapsed, independent of `Revenue` and `PageValues`, retaining the first record. This prevents identical predictors crossing partitions, including rows that differ only in excluded metadata. This conservative policy may discard genuinely distinct sessions and slightly changes the modeling population; the split report records the count. No user IDs or timestamps are invented.

Stratified 70% training, 15% validation and 15% test partitions use seed 42. Integer rounding can shift counts by one. Their original row indices are saved. There is no reliable chronological split because sufficiently granular timestamps are absent. True future-period validation needs new data. Descriptive EDA uses the full sample; automated model selection uses only training and validation. Avoid manually tuning features against test-derived observations.

## Learning and experiment selection

Each learned family is tested on the same partitions with original and engineered features. All imputing, scaling, one-hot encoding and optional screening are fitted inside training folds. Logistic Regression, Decision Tree, Random Forest and SVM use balanced class weights. XGBoost and LightGBM search weighted/unweighted options. All candidates get validation threshold optimization through the selected-model policy. No SMOTE is used.

The default randomized search samples two settings over three stratified folds with Average Precision scoring. The dummy uses class priors as a baseline. Model timings include search/refit; batch latency is amortized over validation rows and is not single-request p95. Estimator threads are bounded for a laptop; outer CV is serial. Training warnings (including convergence warnings if any) are saved beside each candidate. Modest search budgets and early stopping may underfit or disadvantage individual families; comparisons are conditional on this protocol.

Highest validation AP selects family and feature configuration. A predeclared screened variant of that winner uses correlation/MI pruning; it replaces the winner only if validation AP improves. Tree built-in importance and validation permutation importance provide additional diagnostic rankings, not post-hoc test-based selection. Correlation, MI and importance each have biases; no ranking establishes causal drivers.

Calibration fits sigmoid scaling to out-of-fold **training** predictions (`CalibratedClassifierCV`, `ensemble=False`); its final base estimator is refitted on all training data. Validation chooses calibration only if Brier score improves and AP falls by no more than 0.01. This keeps calibration training separate from selection and does not refit on validation. Both calibration results are retained.

The final decision threshold maximizes validation F1 subject to a configured minimum precision (default 0.30). This precision is an observed validation requirement, not a guarantee for future populations. Ties favor precision. The chosen pipeline, calibration and threshold are frozen before test evaluation. Only the selected model receives test metrics; reporting test scores for every model would invite test-based selection.

A single validation set supports several decisions and is therefore subject to selection optimism. A stronger research protocol would use nested CV and a fresh prospective holdout. The test set should not be repeatedly used to guide development. Implementation/debug reruns are stored as separate experiment versions; they are not independent statistical replications.

## Evaluation and uncertainty

AP is prioritized over accuracy for the imbalanced outcome. Reports include ROC-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, accuracy and the fixed-order confusion matrix `[[TN, FP], [FN, TP]]`. AP uses scikit-learn's stepwise definition, not trapezoidal PR area. Calibration and ROC/PR plots come from the final held-out model.

EDA shows 95% Wilson intervals for conversion proportions. Chi-square tests evaluate categorical association with Cramér's V effect sizes and warnings for sparse expected cells. Mann–Whitney U compares numeric distributions, with rank-biserial effect sizes. Bonferroni correction covers the tests shown in the current analysis. Correlated sessions, historical sampling and multiple interactive filter choices can weaken inference; p-values do not establish practical importance or causality.

The cost simulator computes `false_positives × assumed_FP_cost + false_negatives × assumed_FN_cost`. It is an illustrative accounting model, not an estimate of campaign uplift, profit or counterfactual outcomes. Intent segments have configurable boundaries; their labels do not confer certainty.

## Explanations

Permutation importance measures the mean validation AP decrease from shuffling a raw feature (three repeats on a fixed validation subset). Derived features are recomputed by the pipeline after raw-feature shuffling. Correlation can mask or distribute importance.

SHAP uses a model-agnostic permutation explainer over a reversible numeric representation of raw inputs and 24 training background rows. One permutation per explained observation keeps interactive latency manageable; this is an approximate, potentially high-variance attribution. It explains predicted probability even for a calibrated pipeline. A summary uses 24 validation rows and a waterfall uses one validation example. The baseline plus contributions reconstructs the prediction within numerical tolerance, tested automatically.

If SHAP is unavailable, global permutation importance remains available and individual explanations use single-feature background replacement. Those fallback differences are explicitly labeled **non-additive**. Background replacement can combine correlated variables in implausible ways. Neither method proves that increasing a feature causes conversion.
