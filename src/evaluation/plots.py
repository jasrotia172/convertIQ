"""Export reproducible evaluation charts without a GUI."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay


def evaluation_plots(y, probabilities, threshold, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    RocCurveDisplay.from_predictions(y, probabilities, ax=axes[0, 0])
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[0, 1])
    ConfusionMatrixDisplay.from_predictions(y, probabilities >= threshold, ax=axes[1, 0], colorbar=False)
    CalibrationDisplay.from_predictions(y, probabilities, n_bins=10, strategy="quantile", ax=axes[1, 1])
    fig.suptitle("ConvertIQ · Final held-out evaluation")
    fig.tight_layout()
    fig.savefig(directory / "evaluation.png", dpi=150)
    plt.close(fig)
