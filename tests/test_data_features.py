import numpy as np
import pandas as pd
import pytest
from src.data.schema import validate_frame, DataValidationError, FEATURES
from src.features.pipeline import SessionFeatures, preprocessing
from src.models.train import split_dataset
from src.config import settings
from src.data.analytics import analytics, wilson


@pytest.mark.parametrize(
    "column,value",
    [
        ("BounceRates", 2),
        ("ProductRelated", -1),
        ("Browser", 1.5),
        ("Month", "Bad"),
        ("Weekend", "not-a-bool"),
        ("Revenue", "maybe"),
        ("Administrative_Duration", np.inf),
        ("ExitRates", None),
    ],
)
def test_reject_invalid(synthetic_frame, column, value):
    frame = synthetic_frame.copy()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError):
        validate_frame(frame, target=True)


def test_schema_missing_extra(synthetic_frame):
    with pytest.raises(DataValidationError, match="Missing"):
        validate_frame(synthetic_frame.drop(columns="Month"))
    with pytest.raises(DataValidationError, match="unexpected"):
        validate_frame(synthetic_frame.assign(surprise=1))


def test_zero_division_and_no_leakage(synthetic_frame):
    frame = synthetic_frame.copy()
    for col in [
        "Administrative",
        "Informational",
        "ProductRelated",
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "ExitRates",
    ]:
        frame[col] = 0
    transformed = SessionFeatures().fit_transform(frame)
    assert not {"Revenue", "PageValues"} & set(transformed.columns)
    assert np.isfinite(transformed.select_dtypes("number")).all().all()
    assert transformed.DurationPerPage.eq(0).all()


def test_preprocessing_fits_train_only_and_unknown_categories(synthetic_frame):
    train = synthetic_frame.iloc[:60]
    pipe = preprocessing().fit(train)
    mean_before = pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"].mean_.copy()
    inference = synthetic_frame.iloc[60:].copy()
    inference["Browser"] = 999
    inference["ProductRelated_Duration"] = 1e9
    assert np.isfinite(pipe.transform(inference)).all()
    np.testing.assert_array_equal(
        mean_before, pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"].mean_
    )


def test_duplicate_predictors_do_not_cross_splits(synthetic_frame):
    duplicate = synthetic_frame.iloc[[0]].copy()
    duplicate["PageValues"] = 999
    duplicate["Revenue"] = ~duplicate.Revenue
    partitions = split_dataset(pd.concat([synthetic_frame, duplicate], ignore_index=True), settings())
    hashes = [set(pd.util.hash_pandas_object(part[FEATURES], index=False)) for part in partitions]
    assert sum(map(len, partitions)) == len(synthetic_frame)
    assert not hashes[0] & hashes[1] and not hashes[0] & hashes[2] and not hashes[1] & hashes[2]


def test_duckdb_matches_pandas(synthetic_frame):
    result = analytics(synthetic_frame)
    assert result["kpis"]["conversion_rate"] == pytest.approx(synthetic_frame.Revenue.mean())
    assert result["kpis"]["sessions"] == len(synthetic_frame)
    low, high = wilson(15, 100)
    assert low < 0.15 < high


def test_feature_selector_prunes_duplicate_columns():
    from src.features.selection import CorrelationMISelector

    rng = np.random.default_rng(4)
    x = rng.normal(size=150)
    values = np.column_stack([x, x, rng.normal(size=150), np.zeros(150)])
    selector = CorrelationMISelector().fit(values, x > 0)
    assert not ({0, 1} <= set(selector.indices_))
    assert 3 not in selector.indices_
    assert selector.transform(values).shape[1] == len(selector.get_feature_names_out())


def test_all_estimator_parameters_are_strict_json():
    import json
    from src.models.catalog import candidates
    from src.models.train import model_parameters

    for model, _ in candidates().values():
        json.dumps(model_parameters(model), allow_nan=False)
