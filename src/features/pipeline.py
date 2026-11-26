"""Deterministic feature creation and train-fitted transformations."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data.schema import NUMERIC, CATEGORICAL, COUNTS, DURATIONS, MONTHS, FEATURES

ENGINEERED = [
    "TotalPages",
    "TotalDuration",
    "DurationPerPage",
    "ProductPageRatio",
    "ProductDurationRatio",
    "ProductTimePerPage",
    "BounceExitRatio",
    "MonthSin",
    "MonthCos",
]


def safe_divide(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.zeros(len(numerator), dtype=float), where=np.asarray(denominator) != 0
    )


class SessionFeatures(TransformerMixin, BaseEstimator):
    """No target access or fitted statistics; drops leakage-prone metadata."""

    def __init__(self, engineered: bool = True):
        self.engineered = engineered

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X):
        result = X[FEATURES].copy()
        if self.engineered:
            result["TotalPages"] = result[COUNTS].sum(axis=1)
            result["TotalDuration"] = result[DURATIONS].sum(axis=1)
            result["DurationPerPage"] = safe_divide(result.TotalDuration, result.TotalPages)
            result["ProductPageRatio"] = safe_divide(result.ProductRelated, result.TotalPages)
            result["ProductDurationRatio"] = safe_divide(result.ProductRelated_Duration, result.TotalDuration)
            result["ProductTimePerPage"] = safe_divide(result.ProductRelated_Duration, result.ProductRelated)
            result["BounceExitRatio"] = safe_divide(result.BounceRates, result.ExitRates)
            angle = 2 * np.pi * result.Month.map(MONTHS) / 12
            result["MonthSin"], result["MonthCos"] = np.sin(angle), np.cos(angle)
        # Device and source codes are identifiers, not continuous quantities.
        for column in CATEGORICAL:
            result[column] = result[column].astype(str)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES + (ENGINEERED if self.engineered else []), dtype=object)


def preprocessing(engineered: bool = True) -> Pipeline:
    numeric = NUMERIC + (ENGINEERED if engineered else [])
    transforms = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", SessionFeatures(engineered)), ("columns", transforms)])


def model_pipeline(model, engineered: bool = True) -> Pipeline:
    return Pipeline([("preprocess", preprocessing(engineered)), ("model", model)])
