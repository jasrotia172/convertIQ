"""Model-agnostic SHAP in probability units with explicit noncausal fallbacks."""

import logging
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from src.data.schema import FEATURES, CATEGORICAL

LOG = logging.getLogger(__name__)


class ProbabilityExplainer:
    """Encode categories reversibly for SHAP's independent tabular masker.

    This background-replacement explanation can create unlikely feature combinations.
    Contributions describe model behavior, not intervention effects.
    """

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.background = background[FEATURES].head(24).copy()
        self.categories = {c: sorted(self.background[c].unique().tolist(), key=str) for c in CATEGORICAL}

    def encode(self, frame):
        result = frame[FEATURES].copy()
        for col, values in self.categories.items():
            for value in frame[col].unique():
                if value not in values:
                    values.append(value)
            result[col] = result[col].map({value: i for i, value in enumerate(values)})
        return result.to_numpy(dtype=float)

    def predict_encoded(self, values):
        result = pd.DataFrame(values, columns=FEATURES)
        for col, categories in self.categories.items():
            result[col] = [categories[int(np.clip(round(v), 0, len(categories) - 1))] for v in result[col]]
        return self.model.predict_proba(result)[:, 1]

    def explain(self, rows):
        import shap

        background = self.encode(self.background)
        values = self.encode(rows)
        explainer = shap.Explainer(
            self.predict_encoded, background, algorithm="permutation", feature_names=FEATURES, seed=42
        )
        return explainer(values, max_evals=2 * len(FEATURES) + 1, silent=True)

    def individual(self, row: pd.DataFrame) -> dict:
        probability = float(self.model.predict_proba(row[FEATURES])[0, 1])
        try:
            explanation = self.explain(row.iloc[:1])
            return {
                "method": "SHAP permutation (one permutation; approximate)",
                "units": "probability",
                "base_value": float(explanation.base_values[0]),
                "probability": probability,
                "contributions": dict(zip(FEATURES, explanation.values[0].astype(float))),
                "note": "Additive model explanation relative to sampled training background; not causal.",
            }
        except (ImportError, ValueError, TypeError, RuntimeError) as exc:
            LOG.warning("SHAP unavailable; using background replacement: %s", exc)
            contributions = {}
            for column in FEATURES:
                replaced = pd.concat([row.iloc[:1]] * len(self.background), ignore_index=True)
                replaced[column] = self.background[column].values
                contributions[column] = probability - float(self.model.predict_proba(replaced)[:, 1].mean())
            return {
                "method": "single-feature background replacement",
                "units": "probability difference",
                "base_value": None,
                "probability": probability,
                "contributions": contributions,
                "note": "Fallback differences are not additive and are not causal.",
            }


def global_importance(model, X, y, seed=42):
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1
    )
    return pd.DataFrame(
        {"feature": X.columns, "importance": result.importances_mean, "std": result.importances_std}
    ).sort_values("importance", ascending=False)


def export_shap(model, background, rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import shap

        explanation = ProbabilityExplainer(model, background).explain(rows.iloc[:24])
        shap.summary_plot(explanation.values, rows.iloc[:24][FEATURES], show=False)
        plt.tight_layout()
        plt.savefig(directory / "shap_summary.png", dpi=130, bbox_inches="tight")
        plt.close("all")
        # Display original category values, not the explainer's internal code indices.
        original_values = shap.Explanation(
            values=explanation.values[0],
            base_values=explanation.base_values[0],
            data=rows.iloc[0][FEATURES].to_numpy(),
            feature_names=FEATURES,
        )
        shap.plots.waterfall(original_values, show=False)
        plt.savefig(directory / "shap_waterfall.png", dpi=130, bbox_inches="tight")
        plt.close("all")
        return {"available": True, "samples": len(rows.iloc[:24]), "units": "probability"}
    except Exception as exc:
        LOG.warning("SHAP plots unavailable: %s", exc)
        plt.close("all")
        return {"available": False, "reason": str(exc), "fallback": "permutation_importance.csv"}
