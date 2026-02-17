import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def _preprocess_input(raw_features: dict):
    """Apply the same transformations used during training to a single example."""
    df = pd.DataFrame([raw_features])

    # Handle date (be flexible about input format)
    if "date" in df.columns:
        # Try to parse any common date format (e.g. "09/26/2019" or "2024-12-20")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df = df.drop(columns=["date"])

    X = df

    # One-hot encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    # Align to training columns, filling missing columns with 0
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def predict(raw_features: dict) -> float:
    X = _preprocess_input(raw_features)
    return float(model.predict(X)[0])
