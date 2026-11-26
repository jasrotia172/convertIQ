"""Train nine model families; select, calibrate and tune without test access."""

import argparse
from datetime import datetime, timezone
import hashlib
import logging
import os
import time
import uuid
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.features.selection import CorrelationMISelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from threadpoolctl import threadpool_limits
from src.config import ROOT, RAW_FILE, DATA, ARTIFACTS, REPORTS, settings, save_json
from src.data.schema import FEATURES, validate_frame, quality_report
from src.features.pipeline import model_pipeline
from src.models.catalog import candidates
from src.evaluation.metrics import evaluate, choose_threshold, threshold_table
from src.evaluation.plots import evaluation_plots
from src.explainability.explain import global_importance, export_shap

LOG = logging.getLogger(__name__)


def model_parameters(model) -> dict:
    """Keep estimator settings JSON-safe without allowing nonfinite metric values."""
    return {
        key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in model.get_params().items()
    }


def split_dataset(frame, config):
    # Conflicting outcomes for identical inputs remain in EDA but cannot be split safely.
    # Keep the first input vector regardless of label, and report this conservative policy.
    clean = frame.drop_duplicates(subset=FEATURES, keep="first").copy()
    train_val, test = train_test_split(
        clean, test_size=config["test_size"], stratify=clean.Revenue, random_state=config["seed"]
    )
    train, validation = train_test_split(
        train_val,
        test_size=config["validation_size"] / (1 - config["test_size"]),
        stratify=train_val.Revenue,
        random_state=config["seed"],
    )
    return train, validation, test


def train(config_path=None):
    config = settings(config_path)
    seed = config["seed"]
    np.random.seed(seed)
    if not RAW_FILE.exists():
        raise FileNotFoundError("Run python -m src.data.download first.")
    frame = validate_frame(pd.read_csv(RAW_FILE), target=True, raw=True)
    train_df, val_df, test_df = split_dataset(frame, config)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    for name, partition in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        partition.to_csv(DATA / "processed" / f"{name}.csv", index=False)
    dataset_hash = hashlib.sha256(RAW_FILE.read_bytes()).hexdigest()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = ARTIFACTS / "runs" / version
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "reports"
    report_dir.mkdir()
    save_json(report_dir / "config.json", config)
    save_json(report_dir / "quality.json", quality_report(frame))
    save_json(
        report_dir / "split.json",
        {
            "raw_rows": len(frame),
            "removed_identical_inputs": len(frame) - sum(map(len, [train_df, val_df, test_df])),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "policy": "Remove identical predictor rows before stratified splitting; retain first independent of label.",
            "indices": {
                "train": train_df.index.tolist(),
                "validation": val_df.index.tolist(),
                "test": test_df.index.tolist(),
            },
        },
    )
    X, y = train_df[FEATURES], train_df.Revenue.astype(int)
    Xv, yv = val_df[FEATURES], val_df.Revenue.astype(int)
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=seed)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("ConvertIQ")
    trained, rows = {}, []
    # Outer search is serial; estimator threads and BLAS are bounded for laptops.
    for name, (estimator, grid) in candidates(seed, config["n_jobs"]).items():
        for engineered in [False] if name == "dummy" else [False, True]:
            key = f"{name}__{'engineered' if engineered else 'original'}"
            LOG.info("Training %s", key)
            pipe = model_pipeline(clone(estimator), engineered)
            start = time.perf_counter()
            with mlflow.start_run(run_name=f"{version}/{key}") as run:
                with (
                    warnings.catch_warnings(record=True) as captured,
                    threadpool_limits(limits=config["n_jobs"]),
                ):
                    warnings.simplefilter("always")
                    if grid:
                        search = RandomizedSearchCV(
                            pipe,
                            {f"model__{k}": v for k, v in grid.items()},
                            n_iter=config["search_iterations"],
                            scoring="average_precision",
                            cv=cv,
                            refit=True,
                            n_jobs=1,
                            random_state=seed,
                            error_score="raise",
                        )
                        search.fit(X, y)
                        fitted, score, params = (
                            search.best_estimator_,
                            search.best_score_,
                            search.best_params_,
                        )
                        pd.DataFrame(search.cv_results_).to_csv(report_dir / f"cv_{key}.csv", index=False)
                    else:
                        score = float(cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean())
                        fitted, params = pipe.fit(X, y), {}
                duration = time.perf_counter() - start
                start_predict = time.perf_counter()
                probabilities = fitted.predict_proba(Xv)[:, 1]
                latency = (time.perf_counter() - start_predict) * 1000 / len(Xv)
                metrics = evaluate(yv, probabilities)
                row = {
                    "model": key,
                    "family": name,
                    "engineered": engineered,
                    "cv_average_precision": float(score),
                    "training_seconds": duration,
                    "batch_ms_per_row": latency,
                    "mlflow_run_id": run.info.run_id,
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                }
                rows.append(row)
                trained[key] = fitted
                mlflow.log_params(
                    {
                        "dataset_sha256": dataset_hash,
                        "feature_set": "engineered" if engineered else "original",
                        "seed": seed,
                        **params,
                        **{
                            f"estimator__{k}": v
                            for k, v in model_parameters(fitted.named_steps["model"]).items()
                        },
                    }
                )
                mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
                model_path = run_dir / f"{key}.joblib"
                joblib.dump(fitted, model_path, compress=3)
                mlflow.log_artifact(str(model_path), artifact_path="models")
                save_json(
                    report_dir / f"{key}.json",
                    {
                        "search_parameters": params,
                        "parameters": model_parameters(fitted.named_steps["model"]),
                        "validation": metrics,
                        "warnings": sorted(set(str(w.message) for w in captured)),
                    },
                )
                LOG.info("%s validation AP=%.4f, %.1fs", key, metrics["average_precision"], duration)
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    selected = comparison.iloc[0]["model"]
    best = trained[selected]
    # A predeclared train-fitted screening candidate tests whether a smaller model helps.
    screened = Pipeline(
        [
            ("preprocess", clone(best.named_steps["preprocess"])),
            ("select", CorrelationMISelector(random_state=seed)),
            ("model", clone(best.named_steps["model"])),
        ]
    )
    start = time.perf_counter()
    with threadpool_limits(limits=config["n_jobs"]):
        screened_cv = cross_val_score(screened, X, y, cv=cv, scoring="average_precision").mean()
        screened.fit(X, y)
    screen_training_seconds = time.perf_counter() - start
    screen_metrics = evaluate(yv, screened.predict_proba(Xv)[:, 1])
    screen_selected = screen_metrics["average_precision"] > float(comparison.iloc[0]["average_precision"])
    save_json(
        report_dir / "feature_selection.json",
        {
            "method": "Train-fold MI ranking and correlation pruning, retain at most 80%",
            "validation": screen_metrics,
            "cv_average_precision": float(screened_cv),
            "training_seconds": screen_training_seconds,
            "selected": screen_selected,
            "retained_features": screened.named_steps["select"]
            .get_feature_names_out(screened.named_steps["preprocess"].get_feature_names_out())
            .tolist(),
        },
    )
    joblib.dump(screened, run_dir / "screened_candidate.joblib", compress=3)
    with mlflow.start_run(run_name=f"{version}/feature_selection"):
        mlflow.log_params(
            {
                "base_candidate": selected,
                "feature_fraction": 0.8,
                "correlation_limit": 0.95,
                "dataset_sha256": dataset_hash,
                "seed": seed,
            }
        )
        mlflow.log_metrics(
            {f"validation_{k}": v for k, v in screen_metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.log_metric("cv_average_precision", float(screened_cv))
        mlflow.log_artifact(str(report_dir / "feature_selection.json"))
        mlflow.log_artifact(str(run_dir / "screened_candidate.joblib"), artifact_path="models")
    if screen_selected:
        best = screened
        selected += "__screened"
    # Calibrator sees only out-of-fold training predictions. Validation selects calibration.
    calibration = CalibratedClassifierCV(clone(best), method="sigmoid", cv=cv, n_jobs=1, ensemble=False)
    with threadpool_limits(limits=config["n_jobs"]):
        calibration.fit(X, y)
    original_val = evaluate(yv, best.predict_proba(Xv)[:, 1])
    calibrated_val = evaluate(yv, calibration.predict_proba(Xv)[:, 1])
    use_calibration = (
        calibrated_val["brier_score"] < original_val["brier_score"]
        and calibrated_val["average_precision"] >= original_val["average_precision"] - 0.01
    )
    deployed = calibration if use_calibration else best
    val_probabilities = deployed.predict_proba(Xv)[:, 1]
    threshold = choose_threshold(yv, val_probabilities, config["threshold"]["minimum_precision"])
    save_json(
        report_dir / "calibration.json",
        {
            "uncalibrated": original_val,
            "sigmoid_cv": calibrated_val,
            "selected": "sigmoid_cv" if use_calibration else "uncalibrated",
            "rule": "Lower validation Brier score with AP loss <= 0.01.",
        },
    )
    pd.DataFrame(threshold_table(yv, val_probabilities)).to_csv(report_dir / "thresholds.csv", index=False)
    # Inspect selection signals on TRAIN only; explanatory rankings do not change chosen model.
    pre = best.named_steps["preprocess"]
    transformed = pre.transform(X)
    names = pre.get_feature_names_out()
    mi = mutual_info_classif(
        transformed,
        y,
        random_state=seed,
        discrete_features=[n not in pre.named_steps["features"].get_feature_names_out() for n in names],
    )
    pd.DataFrame({"feature": names, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    ).to_csv(report_dir / "mutual_information.csv", index=False)
    numeric_train = pd.DataFrame(transformed, columns=names)
    numeric_train.corr().to_csv(report_dir / "training_correlations.csv")
    model = best.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        pd.DataFrame(
            {
                "feature": best.named_steps["select"].get_feature_names_out(names)
                if "select" in best.named_steps
                else names,
                "importance": model.feature_importances_,
            }
        ).to_csv(report_dir / "model_importance.csv", index=False)
    importance = global_importance(deployed, Xv.iloc[:500], yv.iloc[:500], seed)
    importance.to_csv(report_dir / "permutation_importance.csv", index=False)
    import matplotlib.pyplot as plt

    importance.head(12).sort_values("importance").plot.barh(
        x="feature", y="importance", xerr="std", legend=False, figsize=(8, 6)
    )
    plt.xlabel("Decrease in validation Average Precision")
    plt.tight_layout()
    plt.savefig(report_dir / "feature_importance.png", dpi=150)
    plt.close("all")
    background = X.sample(n=min(100, len(X)), random_state=seed)
    shap_status = export_shap(deployed, background, Xv, report_dir)
    # First and only use of test labels for model evaluation, after all selection decisions.
    test_probabilities = deployed.predict_proba(test_df[FEATURES])[:, 1]
    final_metrics = evaluate(test_df.Revenue, test_probabilities, threshold)
    evaluation_plots(test_df.Revenue, test_probabilities, threshold, report_dir)
    for name, data, probabilities in [
        ("validation", val_df, val_probabilities),
        ("test", test_df, test_probabilities),
    ]:
        pd.DataFrame({"Revenue": data.Revenue.astype(int), "probability": probabilities}).to_csv(
            report_dir / f"{name}_predictions.csv", index=False
        )
    save_json(report_dir / "test_metrics.json", final_metrics)
    metadata = {
        "version": version,
        "selected_model": selected,
        "calibrated": use_calibration,
        "threshold": threshold,
        "segments": config["segments"],
        "dataset_sha256": dataset_hash,
        "features": FEATURES,
        "test_metrics": final_metrics,
        "validation_metrics": evaluate(yv, val_probabilities, threshold),
        "feature_screening": screen_selected,
        "shap": shap_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Session-level propensity; not validated for early-session or causal use.",
    }
    joblib.dump(deployed, run_dir / "pipeline.joblib", compress=3)
    background.to_csv(run_dir / "background.csv", index=False)
    train_df[FEATURES].to_csv(run_dir / "reference.csv", index=False)
    pd.DataFrame({"probability": deployed.predict_proba(X)[:, 1]}).to_csv(
        run_dir / "reference_predictions.csv", index=False
    )
    save_json(run_dir / "metadata.json", metadata)
    with mlflow.start_run(run_name=f"{version}/selected"):
        mlflow.log_params(
            {
                "selected_model": selected,
                "calibrated": use_calibration,
                "threshold": threshold,
                "version": version,
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        mlflow.log_artifact(str(run_dir / "pipeline.joblib"), artifact_path="selected")
    # Publish only complete runs. Existing serving processes reload on pointer change.
    pending = ARTIFACTS / "latest.tmp"
    save_json(pending, {"version": version})
    pending.replace(ARTIFACTS / "latest.json")
    save_json(REPORTS / "latest_run.json", metadata)
    LOG.info("Completed %s. Final test AP %.4f", version, final_metrics["average_precision"])
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    print(train(parser.parse_args().config))
