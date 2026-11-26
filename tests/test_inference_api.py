import numpy as np
import pytest
from fastapi.testclient import TestClient
import api
from src.data.schema import FEATURES
from src.models.schemas import SessionInput
from src.evaluation.metrics import evaluate, choose_threshold


@pytest.fixture
def client(synthetic_predictor, monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: synthetic_predictor)
    api.app.dependency_overrides[api.predictor] = lambda: synthetic_predictor
    with TestClient(api.app) as client:
        yield client
    api.app.dependency_overrides.clear()


def test_inference_consistency(synthetic_predictor, synthetic_frame):
    frame = synthetic_frame[FEATURES].head(3)
    results = synthetic_predictor.predict(frame)
    np.testing.assert_allclose(
        [r["conversion_probability"] for r in results],
        synthetic_predictor.pipeline.predict_proba(frame)[:, 1],
    )
    assert all(r["predicted_class"] == int(r["conversion_probability"] >= 0.4) for r in results)
    with pytest.raises(ValueError):
        synthetic_predictor.predict(frame, threshold=2)
    with pytest.raises(ValueError):
        synthetic_predictor.predict(frame, segments={"medium": 0.8, "high": 0.2})


def test_api_routes(client, synthetic_frame, synthetic_predictor):
    payload = SessionInput(**synthetic_frame[FEATURES].iloc[0].to_dict()).model_dump()
    assert client.get("/health").status_code == 200
    assert client.get("/model/info").json()["version"] == "synthetic-test"
    assert client.get("/model/metrics").status_code == 200
    single = client.post("/predict", json=payload)
    batch = client.post("/predict/batch", json={"sessions": [payload, payload]})
    assert single.status_code == batch.status_code == 200
    assert batch.json()[0]["conversion_probability"] == pytest.approx(single.json()["conversion_probability"])
    assert batch.json()[0]["predicted_class"] == single.json()["predicted_class"]
    assert single.json()["conversion_probability"] == pytest.approx(
        synthetic_predictor.predict(synthetic_frame[FEATURES].iloc[:1])[0]["conversion_probability"]
    )
    assert client.get("/monitoring/status").status_code == 200
    explanation = client.post("/explain", json=payload)
    assert explanation.status_code == 200
    data = explanation.json()
    if data["base_value"] is not None:
        assert data["base_value"] + sum(data["contributions"].values()) == pytest.approx(
            data["probability"], abs=1e-5
        )


@pytest.mark.parametrize(
    "change",
    [{"ProductRelated": -1}, {"Revenue": True}, {"PageValues": 10}, {"Month": "Wrong"}, {"ExitRates": 1.1}],
)
def test_api_validation(client, synthetic_frame, change):
    payload = SessionInput(**synthetic_frame[FEATURES].iloc[0].to_dict()).model_dump()
    payload.update(change)
    assert client.post("/predict", json=payload).status_code == 422


def test_api_empty_batch(client):
    assert client.post("/predict/batch", json={"sessions": []}).status_code == 422


def test_api_no_model(monkeypatch):
    def missing():
        raise FileNotFoundError("No model")

    monkeypatch.setattr(api, "get_predictor", missing)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/model/info").status_code == 503


def test_threshold_optimization():
    y = [0, 0, 0, 1, 1]
    p = [0.01, 0.1, 0.2, 0.3, 0.4]
    threshold = choose_threshold(y, p, minimum_precision=0.5)
    assert evaluate(y, p, threshold)["f1"] == 1
    assert evaluate([0, 0], [0.1, 0.2])["roc_auc"] is None
