"""Integration smoke checks use real artifacts, never synthetic dashboard metrics."""

import pytest
from streamlit.testing.v1 import AppTest
from src.config import ARTIFACTS, ROOT

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "latest.json").exists(), reason="Run training for dashboard integration tests"
)


@pytest.mark.parametrize(
    "page",
    [
        "Executive Overview",
        "Exploratory Data Analysis",
        "Model Performance",
        "Conversion Prediction",
        "Batch Prediction",
        "Model Monitoring",
        "Experiment History",
    ],
)
def test_dashboard_pages(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [error.message for error in app.exception]


def test_dashboard_prediction_and_empty_filter():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Conversion Prediction").run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.metric[0].label == "Purchase probability"
    app.sidebar.radio[0].set_value("Exploratory Data Analysis").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No sessions" in value.value for value in app.info)


def test_batch_csv_workflow(monkeypatch):
    """Exercise the upload parsing/scoring/export path with a real held-out CSV."""
    import io
    import pandas as pd
    import streamlit
    from src.config import DATA
    from src.models.inference import Predictor

    frame = pd.read_csv(DATA / "processed/test.csv").head(5)
    uploaded = io.BytesIO(frame.to_csv(index=False).encode())
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(Predictor, "monitor", lambda *args, **kwargs: {})
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Batch Prediction").run()
    uploaded.seek(0)
    app.button[0].click().run()
    assert not app.exception
    output = app.session_state["batch_result"][2]
    assert len(output) == 5
    assert output.conversion_probability.between(0, 1).all()
    assert set(["model_version", "propensity_segment", "decision_threshold"]) <= set(output)


def test_monitoring_replay_and_synthetic_drift(monkeypatch):
    from src.models.inference import Predictor
    from src.monitoring.drift import monitor

    def no_persistence(self, frame, probabilities=None, latency_ms=None, simulated=False, config=None):
        return monitor(
            self.reference,
            frame,
            self.reference_probabilities,
            probabilities,
            self.metadata["threshold"],
            latency_ms,
            config,
            simulated,
            self.metadata["version"],
        )

    monkeypatch.setattr(Predictor, "monitor", no_persistence)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Model Monitoring").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["monitor_report"]["performance"] is not None
    app.slider[0].set_value(2.0).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["monitor_report"]["performance"] is None
    assert app.session_state["monitor_report"]["simulated"] is True
