# Operations and deployment

## Artifact lifecycle

Training writes an immutable version directory with candidate Joblib pipelines, selected pipeline, background/reference CSVs, metadata, split indices, configuration, CV results, warnings, metrics and charts. `artifacts/latest.json` is replaced atomically only after a complete run. MLflow records parameters, dataset hashes, feature variants, timings, validation metrics, selected test metrics and artifacts in local SQLite plus `mlruns/`.

The API and Streamlit both load `src.models.inference.Predictor`; there is no duplicate prediction implementation. Predictor caching is keyed by the current version pointer. For rollback, stop training, verify the previous complete artifact and replace the pointer with its version using an atomic file replacement. Never load an untrusted Joblib file: deserialization can execute code. Artifact versions require compatible Python and library versions; train in the same environment used for serving.

The default API binds to loopback through the Makefile. Docker containers bind internally to all interfaces, while Compose exposes only localhost. Single-node SQLite is suitable for this portfolio workload; use a managed database, protected object storage and external observability for multiple replicas. Avoid concurrent training invocations into the same output root; run directories are unique but the latest pointer is intentionally last-writer-wins.

## Monitoring

- Quality: missing columns/cells, invalid numeric ranges, invalid types/categories, per-batch invalid rate and examples. Invalid batches can be inspected without scoring them.
- Numeric features and prediction distributions: Population Stability Index with reference-fixed quantile edges, overflow bins and 0.5 count smoothing. Constant references receive a narrow central bin so shifts remain detectable.
- Categorical features: base-2 Jensen–Shannon **divergence** (squared distance), including unseen levels.
- Defaults: PSI > 0.20, JS divergence > 0.10, missing rate > 1%, any invalid rows, and batch inference latency > 500 ms. These are illustrative operational thresholds, not universal significance cutoffs. Configure them in YAML or the dashboard.
- At least 100 records are required for distribution alerts by default. A single API request records latency/quality and returns `insufficient_data` for drift. Single requests are not automatically accumulated into a time window; submit production windows through batch inference or the monitoring CSV workflow.
- Labeled performance floors default to AP 0.30 and recall 0.50 on sufficiently large two-class batches. These configurable operational thresholds are not statistical significance tests. Rejected API requests retain only error types and field paths in a separate SQLite store, never request values.
- Training data provide the reference population. Training prediction distributions can differ from unseen predictions through overfitting; reference-vs-production PSI is a descriptive signal, not a drift hypothesis test with guaranteed false-positive control.
- Labels are optional. No AP, AUC, accuracy or other performance is computed without known, validated `Revenue` outcomes. A single-class batch has unavailable AP/AUC, with other metrics still reported.
- Inference batch latency includes validation and probability computation, excludes monitoring persistence and explanation overhead, and is not total network request latency.

Historical replay uses held-out records and is labeled simulated traffic. Multipliers above 1 change product duration and exit rates; corresponding historical outcomes are removed because they no longer establish performance on altered inputs. Drift simulations are demonstrations, not evidence of observed production deterioration. Uploaded monitoring data can contain malformed inputs and produce a quality-only report.

Monitoring reports persist in `artifacts/monitoring.sqlite` with UTC time, model version, simulation flag, quality, distributions, optional outcomes and thresholds. Simulation and actual uploaded data remain distinguishable in history. The performance chart's time axis is monitoring ingestion time, not invented session timestamps.

## Retraining policy

1. Investigate schema or missing-data alerts first; do not repair a broken collector by retraining.
2. Review sustained drift on sufficiently large batches and relevant segments. Confirm changes against seasonality and product/traffic mix.
3. Join matured purchase outcomes using a real session identifier in the target platform. Inspect precision, recall, AP and calibration against confidence intervals, operational capacity and cost constraints.
4. Retrain a challenger only on approved, point-in-time data. Fit preprocessors and selection inside folds. Use temporal splits once event time is available.
5. Compare the challenger with the incumbent on a fresh future holdout and representative segments. Re-estimate the decision threshold and verify probability calibration.
6. Promote a reviewed artifact, retain rollback capability, monitor latency and quality, and measure intervention effects through randomized experiments.

This repository does not autonomously retrain or overwrite a running model from monitoring uploads. Monitoring is evidence for an investigation, not sufficient authorization or a proof that a replacement is better.

## Troubleshooting

- `libomp.dylib` missing: run `brew install libomp` on macOS. Docker installs `libgomp1`.
- Missing dataset: run `make download` or manual CSV import. Missing model: run `make train`.
- Invalid inputs: use `reports/example_request.json` or the dashboard's downloadable real sample. `June` is the accepted June label; traffic and device codes are integers.
- Different Python/library version when loading models: recreate the recorded environment or retrain under the serving environment. Use `python -m pip check` to verify package resolution.
- Long first SHAP call: Numba can initialize/compile on first use. Subsequent calls are faster; the documented fallback applies when the explainer cannot run.
- Notebook kernel errors: `make notebook` builds a temporary kernel specification for the active interpreter. It does not install a global kernel. A sandbox may need permission to open local kernel sockets.
- Port occupied: use `python -m streamlit run app.py --server.port 8502` or `python -m uvicorn api:app --port 8001`.

Docker training stores its MLflow database and artifacts in the `tracking` and `mlruns` named volumes; local training uses `mlflow.db` and `mlruns/`. `MLFLOW_TRACKING_URI` can override the default tracking destination.
