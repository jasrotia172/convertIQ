"""Train-fitted redundancy filter with mutual-information ranking."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted


class CorrelationMISelector(TransformerMixin, BaseEstimator):
    """Keep the strongest MI feature from each highly correlated neighborhood.

    The retained fraction is predeclared; fitting inside CV prevents label leakage.
    Correlation and MI are screening criteria, not evidence of causal relevance.
    """

    def __init__(self, fraction=0.8, correlation_limit=0.95, random_state=42):
        self.fraction = fraction
        self.correlation_limit = correlation_limit
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        # One-hot inputs are discrete, even after numeric columns have been scaled.
        discrete = np.array([len(np.unique(X[:, i])) <= 2 for i in range(X.shape[1])])
        self.scores_ = mutual_info_classif(X, y, discrete_features=discrete, random_state=self.random_state)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        kept = []
        maximum = max(1, int(np.ceil(X.shape[1] * self.fraction)))
        for index in np.argsort(-self.scores_, kind="stable"):
            if np.var(X[:, index]) == 0:
                continue
            if all(correlations[index, other] < self.correlation_limit for other in kept):
                kept.append(int(index))
            if len(kept) >= maximum:
                break
        self.indices_ = np.sort(kept) if kept else np.array([int(np.argmax(self.scores_))])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        return np.asarray(X)[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "indices_")
        names = np.asarray(
            input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)]
        )
        return names[self.indices_]
