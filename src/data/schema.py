"""Strict schema shared by downloading, training and inference."""

import numpy as np
import pandas as pd

COUNTS = ["Administrative", "Informational", "ProductRelated"]
DURATIONS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
NUMERIC = COUNTS + DURATIONS + ["BounceRates", "ExitRates", "SpecialDay"]
CODES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
CATEGORICAL = ["Month"] + CODES + ["VisitorType", "Weekend"]
FEATURES = NUMERIC + CATEGORICAL
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]


class DataValidationError(ValueError):
    """Actionable data validation failure."""


def boolean_series(series: pd.Series, name: str) -> pd.Series:
    values = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if values.isna().any():
        raise DataValidationError(f"{name}: expected true/false or 1/0; missing values are not allowed.")
    return values.astype(bool)


def validate_frame(frame: pd.DataFrame, target: bool = False, raw: bool = False) -> pd.DataFrame:
    """Normalize valid input; reject missing, nonfinite, negative or malformed values.

    PageValues and Revenue are accepted as dataset metadata but never model inputs.
    Unknown positive device/source codes are allowed and handled by the encoder.
    """
    required = FEATURES + (["Revenue"] if target else []) + (["PageValues"] if raw else [])
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES + ["Revenue", "PageValues"]))
    if missing or extra:
        raise DataValidationError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}.")
    if frame.empty:
        raise DataValidationError("Dataset must contain at least one row.")
    result = frame.copy()
    if result.isna().any().any():
        cols = result.columns[result.isna().any()].tolist()
        raise DataValidationError(f"Missing values in {cols}. Supply complete session aggregates.")
    for name in NUMERIC + CODES + (["PageValues"] if "PageValues" in result else []):
        try:
            result[name] = pd.to_numeric(result[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"{name}: must be numeric.") from exc
        if not np.isfinite(result[name]).all() or (result[name] < 0).any():
            raise DataValidationError(f"{name}: must be finite and nonnegative.")
    for name in COUNTS + CODES:
        if (result[name] % 1 != 0).any() or (name in CODES and (result[name] < 1).any()):
            raise DataValidationError(
                f"{name}: expected nonnegative counts or positive integer category codes."
            )
        result[name] = result[name].astype(int)
    for name in ["BounceRates", "ExitRates", "SpecialDay"]:
        if (result[name] > 1).any():
            raise DataValidationError(f"{name}: expected a proportion between 0 and 1.")
    if not result.Month.isin(MONTHS).all():
        raise DataValidationError(f"Month must be one of {list(MONTHS)}.")
    if not result.VisitorType.isin(VISITORS).all():
        raise DataValidationError(f"VisitorType must be one of {VISITORS}.")
    result["Weekend"] = boolean_series(result.Weekend, "Weekend")
    if "Revenue" in result:
        result["Revenue"] = boolean_series(result.Revenue, "Revenue")
    return result


def quality_report(frame: pd.DataFrame) -> dict:
    """Describe quality without silently deleting outliers or duplicates."""
    outliers = {}
    for column in NUMERIC:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            outliers[column] = int(((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).sum())
    return {
        "rows": len(frame),
        "missing_by_column": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "iqr_outliers": outliers,
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "class_counts": frame.Revenue.value_counts().to_dict() if "Revenue" in frame else {},
    }
