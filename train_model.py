import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def train():

    os.makedirs("model", exist_ok=True)

    df = pd.read_csv("data/crop_dataset.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.drop("Yield_ton_per_hectare", axis=1)
    y = df["Yield_ton_per_hectare"]

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    print("Model trained successfully.")
    print("R2 Score:", round(score, 2))

    # Save model files
    joblib.dump(model, "model/crop_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(X.columns, "model/columns.pkl")

    # Save training statistics
    training_stats = {
        "mean": df.mean(numeric_only=True).to_dict(),
        "std": df.std(numeric_only=True).to_dict()
    }

    joblib.dump(training_stats, "model/training_stats.pkl")

    print("All model files saved successfully.")


if __name__ == "__main__":
    train()
