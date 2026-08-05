"""
modeling.py
===========
CRISP-DM stages: Modeling + Evaluation

- build_regression_model(): predicts a numeric target (e.g. salary)
- build_classification_model(): predicts a categorical target (e.g. job level)

Both use a scikit-learn Pipeline (preprocessing + model) so they can be
saved/loaded as a single object and reused on new data with the same schema.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
)

from config import DatasetConfig


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Auto-detect numeric vs categorical columns and build a preprocessing pipeline."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor


def _prepare_features(df: pd.DataFrame, cfg: DatasetConfig, target_col: str) -> pd.DataFrame:
    """Drop target + configured non-feature columns, return X."""
    drop_cols = set(cfg.drop_for_model) | {target_col}
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)


def build_regression_model(df: pd.DataFrame, cfg: DatasetConfig) -> dict:
    """Train + evaluate a RandomForestRegressor to predict cfg.regression_target."""
    target = cfg.regression_target
    X = _prepare_features(df, cfg, target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    preprocessor = _build_preprocessor(X)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150, max_depth=20, n_jobs=-1, random_state=cfg.random_state
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }
    print(f"[build_regression_model] Metrics: {metrics}")

    os.makedirs(cfg.model_dir, exist_ok=True)
    path = os.path.join(cfg.model_dir, "regression_model.joblib")
    joblib.dump(model, path)
    print(f"[build_regression_model] Saved model to {path}")

    return {"model": model, "metrics": metrics, "model_path": path}


def build_classification_model(df: pd.DataFrame, cfg: DatasetConfig) -> dict:
    """Train + evaluate a RandomForestClassifier to predict cfg.classification_target."""
    target = cfg.classification_target
    X = _prepare_features(df, cfg, target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    preprocessor = _build_preprocessor(X)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=20, n_jobs=-1, random_state=cfg.random_state
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    report = classification_report(y_test, y_pred)
    print(f"[build_classification_model] Metrics: {metrics}")
    print(report)

    os.makedirs(cfg.model_dir, exist_ok=True)
    path = os.path.join(cfg.model_dir, "classification_model.joblib")
    joblib.dump(model, path)
    print(f"[build_classification_model] Saved model to {path}")

    return {"model": model, "metrics": metrics, "report": report, "model_path": path}


def load_model(path: str):
    """Load a previously saved model (regression or classification)."""
    return joblib.load(path)


def predict_new(model, new_df: pd.DataFrame) -> np.ndarray:
    """Run predictions on new data with the same feature schema used in training."""
    return model.predict(new_df)


if __name__ == "__main__":
    from config import CONFIG
    from data_prep import run_data_prep

    df = run_data_prep(CONFIG)
    reg_results = build_regression_model(df, CONFIG)
    clf_results = build_classification_model(df, CONFIG)
