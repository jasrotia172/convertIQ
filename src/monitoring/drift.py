"""Reference-fixed distribution monitoring and explicitly labeled simulations."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from src.config import ARTIFACTS, settings
from src.data.schema import FEATURES, NUMERIC, CATEGORICAL, validate_frame, DataValidationError
from src.evaluation.metrics import evaluate


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return 0.0
    # Preserve constant reference shifts: a central point bucket with two tail bins.
    cuts = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cuts) == 1:
        delta = max(abs(cuts[0]) * 1e-6, 1e-6)
        edges = np.array([-np.inf, cuts[0] - delta, cuts[0] + delta, np.inf])
    else:
        edges = np.r_[-np.inf, cuts, np.inf]
    a = np.histogram(ref, bins=edges)[0].astype(float) + 0.5
    b = np.histogram(cur, bins=edges)[0].astype(float) + 0.5
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def categorical_js(reference, current) -> float:
    a, b = pd.Series(reference).astype(str).value_counts(), pd.Series(current).astype(str).value_counts()
    values = a.index.union(b.index)
    pa = a.reindex(values, fill_value=0).to_numpy(dtype=float) + 0.5
    pb = b.reindex(values, fill_value=0).to_numpy(dtype=float) + 0.5
    return float(jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2) ** 2)


def quality_summary(current: pd.DataFrame) -> dict:
    missing_columns = sorted(set(FEATURES) - set(current.columns))
    invalid_rows = 0
    errors = []
    try:
        validate_frame(current)
        valid_batch = True
    except DataValidationError:
        valid_batch = False
    for index in [] if valid_batch else range(len(current)):
        try:
            validate_frame(current.iloc[[index]])
        except DataValidationError as exc:
            invalid_rows += 1
            if len(errors) < 5:
                errors.append(str(exc))
    missing_rate = float(current.reindex(columns=FEATURES).isna().to_numpy().mean()) if len(current) else 1.0
    return {
        "rows": len(current),
        "missing_columns": missing_columns,
        "missing_rate": missing_rate,
        "invalid_rows": invalid_rows,
        "invalid_rate": invalid_rows / max(len(current), 1),
        "examples": errors,
    }


def monitor(
    reference,
    current,
    reference_probabilities=None,
    probabilities=None,
    threshold=0.5,
    latency_ms=None,
    config=None,
    simulated=False,
    version=None,
):
    cfg = config or settings()["monitoring"]
    quality = quality_summary(current)
    alerts = []
    if quality["missing_rate"] > cfg["missing_alert"]:
        alerts.append("Missing data exceeds threshold")
    if quality["invalid_rate"] > cfg["invalid_alert"] or quality["missing_columns"]:
        alerts.append("Invalid input detected")
    sufficient = len(current) >= cfg["minimum_batch"]
    numeric, categorical = {}, {}
    if sufficient and not quality["invalid_rows"] and not quality["missing_columns"]:
        clean = validate_frame(current)
        numeric = {c: psi(reference[c], clean[c]) for c in NUMERIC}
        categorical = {c: categorical_js(reference[c], clean[c]) for c in CATEGORICAL}
        alerts += [f"Numerical drift: {c}" for c, value in numeric.items() if value > cfg["psi_alert"]]
        alerts += [f"Categorical drift: {c}" for c, value in categorical.items() if value > cfg["js_alert"]]
    prediction_psi = None
    if sufficient and probabilities is not None and reference_probabilities is not None:
        prediction_psi = psi(reference_probabilities, probabilities)
        if prediction_psi > cfg["psi_alert"]:
            alerts.append("Prediction distribution drift")
    if latency_ms is not None and latency_ms > cfg["latency_ms_alert"]:
        alerts.append("Inference batch latency exceeds threshold")
    performance = None
    if "Revenue" in current and probabilities is not None and not quality["invalid_rows"]:
        clean = validate_frame(current)
        performance = evaluate(clean.Revenue.astype(int), probabilities, threshold)
        if sufficient and performance["average_precision"] is not None:
            if performance["average_precision"] < cfg.get("minimum_average_precision", 0):
                alerts.append("Labeled batch Average Precision below configured floor")
            if performance["recall"] < cfg.get("minimum_recall", 0):
                alerts.append("Labeled batch recall below configured floor")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "simulated": simulated,
        "quality": quality,
        "sufficient_batch": sufficient,
        "numeric_psi": numeric,
        "categorical_js": categorical,
        "prediction_psi": prediction_psi,
        "latency_ms": latency_ms,
        "performance": performance,
        "alerts": alerts,
        "thresholds": cfg,
        "status": "alert" if alerts else ("ok" if sufficient else "insufficient_data"),
    }


def save_monitoring(report, directory: Path = ARTIFACTS):
    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(directory / "monitoring.sqlite", timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, report TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO batches(report) VALUES (?)", (json.dumps(report, allow_nan=False),))


def monitoring_history(directory: Path = ARTIFACTS, limit=100):
    path = directory / "monitoring.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=30) as connection:
        return [
            json.loads(row[0])
            for row in connection.execute("SELECT report FROM batches ORDER BY id DESC LIMIT ?", (limit,))
        ]


def simulate(frame, multiplier: float = 1.0):
    result = frame.copy()
    if multiplier != 1:
        result["ProductRelated_Duration"] *= multiplier
        result["ExitRates"] = (result.ExitRates * multiplier).clip(0, 1)
        # Original labels no longer describe changed covariates; never score synthetic drift.
        result = result.drop(columns=["Revenue"], errors="ignore")
    return result


def record_rejection(endpoint: str, errors: list[dict], directory: Path = ARTIFACTS) -> None:
    """Retain validation error types and field paths, never rejected payload values."""
    directory.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "errors": [{"type": str(e.get("type")), "field": list(e.get("loc", []))} for e in errors],
    }
    with sqlite3.connect(directory / "rejections.sqlite", timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS rejections (id INTEGER PRIMARY KEY, event TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO rejections(event) VALUES (?)", (json.dumps(event),))


def rejection_summary(directory: Path = ARTIFACTS) -> dict:
    path = directory / "rejections.sqlite"
    if not path.exists():
        return {"count": 0, "recent": []}
    with sqlite3.connect(path, timeout=30) as connection:
        count = connection.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
        recent = [
            json.loads(row[0])
            for row in connection.execute("SELECT event FROM rejections ORDER BY id DESC LIMIT 20")
        ]
    return {"count": count, "recent": recent}
