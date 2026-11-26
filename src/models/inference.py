"""Single inference contract shared by the dashboard and API."""

from functools import lru_cache
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from src.config import ARTIFACTS
from src.data.schema import FEATURES, validate_frame
from src.explainability.explain import ProbabilityExplainer
from src.monitoring.drift import monitor, save_monitoring


class Predictor:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.metadata = json.loads((self.run_dir / "metadata.json").read_text())
        # Only trusted, locally produced model artifacts may be loaded.
        self.pipeline = joblib.load(self.run_dir / "pipeline.joblib")
        self.reference = validate_frame(pd.read_csv(self.run_dir / "reference.csv"))
        self.background = validate_frame(pd.read_csv(self.run_dir / "background.csv"))
        self.reference_probabilities = pd.read_csv(
            self.run_dir / "reference_predictions.csv"
        ).probability.values

    def predict(self, frame, threshold=None, segments=None, record=False):
        start = time.perf_counter()
        clean = validate_frame(frame)
        threshold = self.metadata["threshold"] if threshold is None else threshold
        segments = self.metadata["segments"] if segments is None else segments
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between zero and one.")
        if not 0 <= segments["medium"] < segments["high"] <= 1:
            raise ValueError("Segment boundaries must satisfy 0 <= medium < high <= 1.")
        probabilities = self.pipeline.predict_proba(clean[FEATURES])[:, 1]
        results = [
            {
                "predicted_class": int(p >= threshold),
                "conversion_probability": float(p),
                "propensity_segment": "high"
                if p >= segments["high"]
                else "medium"
                if p >= segments["medium"]
                else "low",
                "model_version": self.metadata["version"],
                "decision_threshold": float(threshold),
            }
            for p in probabilities
        ]
        latency = (time.perf_counter() - start) * 1000
        if record:
            self.monitor(clean, probabilities, latency, threshold=threshold)
        return results

    def explain(self, frame):
        clean = validate_frame(frame)
        return ProbabilityExplainer(self.pipeline, self.background).individual(clean.iloc[:1])

    def monitor(
        self, frame, probabilities=None, latency_ms=None, simulated=False, config=None, threshold=None
    ):
        result = monitor(
            self.reference,
            frame,
            self.reference_probabilities,
            probabilities,
            self.metadata["threshold"] if threshold is None else threshold,
            latency_ms,
            config,
            simulated,
            self.metadata["version"],
        )
        save_monitoring(result, self.run_dir.parents[1])
        return result


@lru_cache(maxsize=2)
def _load(version: str, root: str):
    return Predictor(Path(root) / "runs" / version)


def get_predictor():
    pointer = ARTIFACTS / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("No trained model. Run python -m src.models.train first.")
    version = json.loads(pointer.read_text())["version"]
    return _load(version, str(ARTIFACTS))
