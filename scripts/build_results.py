"""Render a human-readable report exclusively from real saved artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from src.config import REPORTS
from src.models.inference import get_predictor


def table(frame):
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(rows)


model = get_predictor()
metadata = model.metadata
directory = model.run_dir / "reports"
comparison = pd.read_csv(directory / "model_comparison.csv")
split = json.loads((directory / "split.json").read_text())
selection = json.loads((directory / "feature_selection.json").read_text())
findings = json.loads((REPORTS / "eda/findings.json").read_text())
importance = pd.read_csv(directory / "permutation_importance.csv")
metrics = pd.DataFrame(
    [(k, v) for k, v in metadata["test_metrics"].items() if k != "confusion_matrix"],
    columns=["Metric", "Held-out test"],
)
ablation = (
    comparison.pivot(index="family", columns="engineered", values="average_precision")
    .rename(columns={False: "Original AP", True: "Engineered AP"})
    .dropna()
)
ablation["AP change"] = ablation["Engineered AP"] - ablation["Original AP"]
text = (
    f"""# ConvertIQ — actual execution results

Generated from model version **{metadata["version"]}**. Dataset SHA-256: `{metadata["dataset_sha256"]}`.

## Observed business patterns

"""
    + "\n".join("- " + s for s in findings["observations"])
    + f"""

These are historical associations; traffic codes are not named marketing channels. Interval and test details are in `reports/eda/`.

## Reproducible partition

{split["raw_rows"]:,} raw rows; {split["removed_identical_inputs"]:,} identical predictor rows removed for modeling. Train **{split["train"]:,}**, validation **{split["validation"]:,}**, test **{split["test"]:,}**. Raw EDA retains all records.

## Validation comparison

At threshold 0.5; selection prioritizes AP, not accuracy. Batch latency is amortized milliseconds per row, not an individual-request tail-latency benchmark.

"""
    + table(
        comparison[
            [
                "model",
                "cv_average_precision",
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "training_seconds",
                "batch_ms_per_row",
            ]
        ]
    )
    + """

## Feature-engineering ablation

"""
    + table(ablation.reset_index())
    + f"""

Additional train-only correlation/MI screening retained {len(selection["retained_features"])} encoded columns. Validation AP: **{selection["validation"]["average_precision"]:.4f}**; selected: **{selection["selected"]}**. This candidate is separate from the original/engineered table above.

## Selected pipeline and final evaluation

Selected: **{metadata["selected_model"]}**. Sigmoid calibration selected: **{metadata["calibrated"]}**. Decision threshold: **{metadata["threshold"]:.2f}**, selected using validation F1 subject to minimum precision 0.30. No post-selection refit used validation or test rows.

"""
    + table(metrics)
    + f"""

Confusion matrix `[[TN, FP], [FN, TP]]`: `{metadata["test_metrics"]["confusion_matrix"]}`.

The test split supports final reporting only. Stored versions from development reruns are not independent replications. The historical sample, modest search budget and single held-out split limit generalization; no early-session or causal claim is justified.

## Global model behavior

Largest raw-feature permutation importance values on the fixed validation subset (mean decrease in AP):

"""
    + table(importance.head(8))
    + f"""

Importance is conditional on this fitted model and correlations between inputs. It is not an intervention effect. SHAP summary/waterfall available: **{metadata["shap"]["available"]}**.

## Visual evidence

![Monthly and visitor conversion](eda/conversion_segments.png)

![Engagement distributions](eda/engagement.png)

Final evaluation and explanation PNGs are in `{model.run_dir.relative_to(REPORTS.parent)}/reports/`. Open the dashboard's Model Performance page for interactive views.
"""
)
(REPORTS / "RESULTS.md").write_text(text)
print(REPORTS / "RESULTS.md")
