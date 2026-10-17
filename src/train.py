from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(path="config/config.yaml"):
    """Load the YAML configuration from the given path."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Le fichier de configuration est introuvable : {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Le fichier de configuration doit contenir un dictionnaire YAML valide."
        )
    return config


def load_dataset():
    """Load the Breast Cancer Wisconsin dataset."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def validate_dataset(X, y):
    """Validate feature and target data before training."""
    if X.empty:
        raise ValueError("Le jeu de données ne contient aucune observation.")
    if len(X) != len(y):
        raise ValueError("Le nombre d'observations et de cibles diffère.")
    if X.isnull().any().any():
        raise ValueError("Des valeurs manquantes ont été détectées dans les variables.")
    if y.isnull().any():
        raise ValueError("Des valeurs manquantes ont été détectées dans la cible.")
    if y.nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")
    if not all(dtype.kind in "iuf" for dtype in X.dtypes):
        raise TypeError("Toutes les variables doivent être numériques.")
    return True


def build_pipeline(config):
    """Create a preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config["model"]["C"],
                    max_iter=config["model"]["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def split_data(X, y, config):
    """Split data into stratified training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=y,
    )


def train_and_evaluate(X_train, X_test, y_train, y_test, config):
    """Train the pipeline and calculate evaluation metrics."""
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return pipeline, metrics


def run_training(config_path="config/config.yaml"):
    """Run a reproducible training experiment and log it to MLflow."""
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X, y = load_dataset()
    validate_dataset(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": config["seed"],
                "test_size": config["data"]["test_size"],
                "C": config["model"]["C"],
                "max_iter": config["model"]["max_iter"],
                "dataset": "Breast Cancer Wisconsin (scikit-learn)",
            }
        )

        pipeline, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test, config
        )

        mlflow.log_metrics(metrics)
        # artifact_path is retained for compatibility with the TP instructions.
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print("Métriques :", metrics)

    return pipeline, metrics


if __name__ == "__main__":
    run_training()
