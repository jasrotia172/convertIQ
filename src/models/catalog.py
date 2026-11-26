"""CPU model registry and compact, reproducible search spaces."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def candidates(seed: int = 42, jobs: int = 2) -> dict:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    return {
        "dummy": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
            {"C": [0.1, 1, 10]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"max_depth": [4, 7, 12], "min_samples_leaf": [10, 30, 60]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=180, class_weight="balanced", n_jobs=jobs, random_state=seed),
            {"max_depth": [8, 16, None], "min_samples_leaf": [3, 10]},
        ),
        "svm": (
            SVC(probability=True, class_weight="balanced", cache_size=512, random_state=seed),
            {"C": [0.3, 1, 3], "gamma": ["scale", 0.01]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"n_estimators": [100, 180], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=180, tree_method="hist", n_jobs=jobs, eval_metric="logloss", random_state=seed
            ),
            {"max_depth": [3, 5], "learning_rate": [0.03, 0.08], "scale_pos_weight": [1, 5]},
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=180,
                verbosity=-1,
                n_jobs=jobs,
                random_state=seed,
                deterministic=True,
                force_col_wise=True,
            ),
            {"num_leaves": [15, 31], "learning_rate": [0.03, 0.08], "class_weight": [None, "balanced"]},
        ),
        "mlp": (
            MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=180, early_stopping=True, random_state=seed),
            {"alpha": [0.001, 0.01], "learning_rate_init": [0.001, 0.003]},
        ),
    }
