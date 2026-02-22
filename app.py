import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
MODEL_PATH = "model/crop_model.pkl"
SCALER_PATH = "model/scaler.pkl"
COLUMNS_PATH = "model/columns.pkl"
STATS_PATH = "model/training_stats.pkl"

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
crop_df = pd.read_csv("data/crop_dataset.csv")
advisory_df = pd.read_csv("data/advisory_knowledge.csv")
market_df = pd.read_csv("data/market_prices.csv")

# 🔥 TAKE VALUES DIRECTLY FROM DATASET
states_list = sorted(crop_df["State"].unique())
crops_list = sorted(crop_df["Crop"].unique())
seasons_list = sorted(crop_df["Season"].unique())

# --------------------------------------------------
# AUTO TRAIN IF MODEL FILES MISSING
# --------------------------------------------------
def train_model():

    print("Training model automatically...")
    os.makedirs("model", exist_ok=True)

    df = crop_df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.drop("Yield_ton_per_hectare", axis=1)
    y = df["Yield_ton_per_hectare"]

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns, COLUMNS_PATH)

    training_stats = {
        "mean": df.mean(numeric_only=True).to_dict(),
        "std": df.std(numeric_only=True).to_dict()
    }

    joblib.dump(training_stats, STATS_PATH)

    print("Model trained successfully.")


if not (
    os.path.exists(MODEL_PATH) and
    os.path.exists(SCALER_PATH) and
    os.path.exists(COLUMNS_PATH) and
    os.path.exists(STATS_PATH)
):
    train_model()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)
training_stats = joblib.load(STATS_PATH)

# --------------------------------------------------
# PREPARE INPUT
# --------------------------------------------------
def prepare_input(form_data):

    input_data = {
        "Area_hectares": float(form_data["area"]),
        "Rainfall_mm": float(form_data["rainfall"]),
        "Temperature_c": float(form_data["temperature"]),
        "Humidity_percent": float(form_data["humidity"]),
        "Nitrogen": float(form_data["nitrogen"]),
        "Phosphorus": float(form_data["phosphorus"]),
        "Potassium": float(form_data["potassium"]),
        "Soil_pH": float(form_data["soil_ph"])
    }

    state = form_data["state"]
    crop = form_data["crop"]
    season = form_data["season"]

    for col in columns:
        if col.startswith("State_"):
            input_data[col] = 1 if col == f"State_{state}" else 0
        elif col.startswith("Crop_"):
            input_data[col] = 1 if col == f"Crop_{crop}" else 0
        elif col.startswith("Season_"):
            input_data[col] = 1 if col == f"Season_{season}" else 0

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=columns, fill_value=0)

    scaled_input = scaler.transform(df)

    return scaled_input, df

# --------------------------------------------------
# ADVISORY ENGINE
# --------------------------------------------------
def generate_advice(input_df, crop):

    advice_list = []

    rainfall = input_df["Rainfall_mm"].values[0]
    nitrogen = input_df["Nitrogen"].values[0]
    phosphorus = input_df["Phosphorus"].values[0]
    potassium = input_df["Potassium"].values[0]
    soil_ph = input_df["Soil_pH"].values[0]
    temperature = input_df["Temperature_c"].values[0]

    conditions = []

    if nitrogen < training_stats["mean"]["Nitrogen"]:
        conditions.append("Low_Nitrogen")

    if phosphorus < training_stats["mean"]["Phosphorus"]:
        conditions.append("Low_Phosphorus")

    if potassium < training_stats["mean"]["Potassium"]:
        conditions.append("Low_Potassium")

    if rainfall < training_stats["mean"]["Rainfall_mm"]:
        conditions.append("Low_Rainfall")

    if temperature > 38:
        conditions.append("High_Temperature")

    if soil_ph < 5.5:
        conditions.append("Low_Soil_pH")

    if soil_ph > 7.5:
        conditions.append("High_Soil_pH")

    for condition in conditions:
        result = advisory_df[
            ((advisory_df["Crop"] == crop) | (advisory_df["Crop"] == "General")) &
            (advisory_df["Condition"] == condition)
        ]

        for _, row in result.iterrows():
            advice_list.append(row["Advice"])

    if not advice_list:
        advice_list.append("Conditions are optimal. Maintain current practices.")

    return advice_list

# --------------------------------------------------
# PROFIT CALCULATION
# --------------------------------------------------
def calculate_profit(predicted_yield, area, crop):

    price_row = market_df[market_df["Crop"] == crop]

    if not price_row.empty:
        price_per_ton = price_row["Price_per_ton"].values[0]
        revenue = predicted_yield * area * price_per_ton
        estimated_cost = area * 10000
        profit = revenue - estimated_cost
        return revenue, profit

    return None, None

# --------------------------------------------------
# ROUTE
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    advice = None
    revenue = None
    profit = None

    if request.method == "POST":

        prepared_data, input_df = prepare_input(request.form)

        prediction = round(model.predict(prepared_data)[0], 2)

        crop = request.form["crop"]
        area = float(request.form["area"])

        advice = generate_advice(input_df, crop)
        revenue, profit = calculate_profit(prediction, area, crop)

    return render_template(
        "index.html",
        prediction=prediction,
        advice=advice,
        revenue=revenue,
        profit=profit,
        states=states_list,
        crops=crops_list,
        seasons=seasons_list
    )

if __name__ == "__main__":
    app.run(debug=True)
