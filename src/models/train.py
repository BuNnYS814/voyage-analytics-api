import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

mlflow.set_experiment("Voyage_Analytics_Pro")

def train():
    # Load the main flights dataset for price prediction
    df = pd.read_csv("data/raw/flights.csv")

    # Basic cleaning
    df = df.dropna()

    # Extract useful numeric features from the date and drop the raw string column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df = df.drop(columns=["date"])

    # Separate target
    X = df.drop("price", axis=1)
    y = df["price"]

    # One-hot encode categorical (object) columns so XGBoost gets only numeric inputs
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    # Save the column order so inference can build features in the same way
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_metric("mse", mse)
        mlflow.xgboost.log_model(model, "model")

        # Optionally register the model in the MLflow Model Registry (local by default)
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="Voyage_Analytics_Pro_Model")
        except Exception:
            # Registration can fail if registry is not configured; training should still succeed.
            pass

        # Persist model and feature metadata for inference
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        joblib.dump(feature_columns, "models/feature_columns.pkl")

if __name__ == "__main__":
    train()
