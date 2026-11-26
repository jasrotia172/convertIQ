"""Explicitly synthetic fixtures: no fixture metrics are presented as dataset results."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.features.pipeline import model_pipeline


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(71)
    count = 100
    return pd.DataFrame(
        {
            "Administrative": rng.integers(0, 5, count),
            "Administrative_Duration": rng.uniform(0, 100, count),
            "Informational": rng.integers(0, 3, count),
            "Informational_Duration": rng.uniform(0, 80, count),
            "ProductRelated": rng.integers(0, 30, count),
            "ProductRelated_Duration": rng.uniform(0, 1000, count),
            "BounceRates": rng.uniform(0, 0.2, count),
            "ExitRates": rng.uniform(0, 0.2, count),
            "SpecialDay": np.zeros(count),
            "Month": rng.choice(["May", "Nov"], count),
            "OperatingSystems": rng.integers(1, 4, count),
            "Browser": rng.integers(1, 4, count),
            "Region": rng.integers(1, 4, count),
            "TrafficType": rng.integers(1, 4, count),
            "VisitorType": rng.choice(["New_Visitor", "Returning_Visitor"], count),
            "Weekend": rng.choice([True, False], count),
            "PageValues": rng.uniform(0, 50, count),
            "Revenue": rng.choice([True, False], count),
        }
    )


@pytest.fixture
def synthetic_predictor(synthetic_frame, tmp_path):
    import joblib
    from src.config import save_json
    from src.data.schema import FEATURES
    from src.models.inference import Predictor

    frame = synthetic_frame
    directory = tmp_path / "runs" / "synthetic-test"
    directory.mkdir(parents=True)
    pipe = model_pipeline(LogisticRegression(max_iter=500)).fit(frame[FEATURES], frame.Revenue)
    joblib.dump(pipe, directory / "pipeline.joblib")
    save_json(
        directory / "metadata.json",
        {
            "version": "synthetic-test",
            "threshold": 0.4,
            "segments": {"medium": 0.2, "high": 0.6},
            "test_metrics": {},
            "validation_metrics": {},
        },
    )
    for name in ["reference", "background"]:
        frame[FEATURES].head(24).to_csv(directory / f"{name}.csv", index=False)
    pd.DataFrame({"probability": pipe.predict_proba(frame[FEATURES].head(24))[:, 1]}).to_csv(
        directory / "reference_predictions.csv", index=False
    )
    return Predictor(directory)
