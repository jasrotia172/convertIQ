"""Record an explicitly simulated historical replay and controlled drift batch."""

import argparse
import time
import pandas as pd
from src.config import DATA
from src.models.inference import get_predictor
from src.monitoring.drift import simulate


def run_demo(multiplier: float = 2.0) -> list[dict]:
    model = get_predictor()
    historical = pd.read_csv(DATA / "processed/test.csv")
    reports = []
    for frame in [
        historical.iloc[:600],
        historical.iloc[600:1200],
        simulate(historical.iloc[1200:], multiplier),
    ]:
        start = time.perf_counter()
        predictions = model.predict(frame)
        latency = (time.perf_counter() - start) * 1000
        reports.append(
            model.monitor(
                frame, [row["conversion_probability"] for row in predictions], latency, simulated=True
            )
        )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=float, default=2.0)
    multiplier = parser.parse_args().multiplier
    if multiplier < 1:
        parser.error("Use a multiplier of at least 1.0.")
    for report in run_demo(multiplier):
        print(
            {
                "simulated": report["simulated"],
                "rows": report["quality"]["rows"],
                "status": report["status"],
                "labels_available": report["performance"] is not None,
                "alerts": report["alerts"],
            }
        )
