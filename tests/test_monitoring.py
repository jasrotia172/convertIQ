import numpy as np
import pytest
from src.monitoring.drift import psi, categorical_js, monitor, simulate, save_monitoring, monitoring_history
from src.data.schema import FEATURES


def test_psi_identity_shift_and_constant():
    values = np.linspace(0, 1, 1000)
    assert psi(values, values) == pytest.approx(0)
    assert psi(values, values + 3) > 0.2
    assert psi(np.zeros(100), np.ones(100)) > 0.2


def test_categorical_unseen():
    assert categorical_js(["a"] * 100, ["a"] * 100) == pytest.approx(0)
    assert categorical_js(["a"] * 100, ["b"] * 100) > 0.5


def test_ground_truth_required(synthetic_frame):
    ref = synthetic_frame[FEATURES]
    unlabeled = monitor(ref, ref, probabilities=np.full(len(ref), 0.4))
    assert unlabeled["performance"] is None
    labeled = monitor(ref, synthetic_frame, probabilities=np.full(len(ref), 0.4))
    assert labeled["performance"]["brier_score"] is not None


def test_invalid_quality_and_small_batch(synthetic_frame):
    invalid = synthetic_frame.head(3).copy()
    invalid["BounceRates"] = -1
    report = monitor(synthetic_frame, invalid)
    assert report["quality"]["invalid_rows"] == 3
    assert report["status"] == "alert"
    assert not report["numeric_psi"]
    missing = monitor(synthetic_frame, invalid.drop(columns="Month"))
    assert "Month" in missing["quality"]["missing_columns"]


def test_synthetic_drift_drops_outcomes(synthetic_frame):
    assert "Revenue" not in simulate(synthetic_frame, 2)
    assert "Revenue" in simulate(synthetic_frame, 1)


def test_history(synthetic_frame, tmp_path):
    report = monitor(synthetic_frame, synthetic_frame, simulated=True)
    save_monitoring(report, tmp_path)
    assert monitoring_history(tmp_path)[0] == report


def test_labeled_performance_alert(synthetic_frame):
    from src.config import settings

    config = settings()["monitoring"]
    config["minimum_average_precision"] = 0.99
    report = monitor(
        synthetic_frame,
        synthetic_frame,
        probabilities=1 - synthetic_frame.Revenue.astype(float).values,
        config=config,
    )
    assert any("Average Precision" in message for message in report["alerts"])
    assert any("recall" in message for message in report["alerts"])


def test_rejection_log_has_no_payload(tmp_path):
    from src.monitoring.drift import record_rejection, rejection_summary

    record_rejection(
        "/predict", [{"type": "bad", "loc": ["body", "Month"], "input": "sensitive-value"}], tmp_path
    )
    result = rejection_summary(tmp_path)
    assert result["count"] == 1
    assert "sensitive-value" not in str(result)
