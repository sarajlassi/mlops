import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.train import (
    build_pipeline,
    load_config,
    validate_dataset,
)


def test_dataset_is_valid():
    dataset = load_breast_cancer(as_frame=True)
    assert validate_dataset(dataset.data, dataset.target) is True


def test_pipeline_can_fit_and_predict():
    config = load_config()
    dataset = load_breast_cancer(as_frame=True)

    X_train, X_test, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=dataset.target,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset(set(dataset.target.unique()))


def test_validation_rejects_empty_dataset():
    X = pd.DataFrame()
    y = pd.Series(dtype="int64")

    with pytest.raises(ValueError, match="aucune observation"):
        validate_dataset(X, y)


def test_validation_rejects_missing_values():
    X = pd.DataFrame({"feature": [1.0, None, 3.0]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="valeurs manquantes"):
        validate_dataset(X, y)
