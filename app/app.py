from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make the project root available to python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prediction import predict_delivery_minutes


MODEL_PATH = ROOT / "models" / "delivery_model.pkl"
CONFIG_PATH = ROOT / "models" / "model_config.pkl"
DATA_PATH = ROOT / "data" / "processed" / "model_ready_data.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "test_predictions.csv"

st.set_page_config(page_title="DeliveryETA", layout="wide")
st.title("DeliveryETA")
st.write("Predict the estimated delivery time in minutes.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

if CONFIG_PATH.exists():
    config = joblib.load(CONFIG_PATH)
else:
    config = {}

data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
test_predictions = (
    pd.read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else None
)

fallback_categories = {
    "Weatherconditions": ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"],
    "Road_traffic_density": ["High", "Jam", "Low", "Medium"],
    "Type_of_order": ["Buffet", "Drinks", "Meal", "Snack"],
    "Type_of_vehicle": ["electric_scooter", "motorcycle", "scooter"],
    "Festival": ["No", "Yes"],
    "City": ["Metropolitian", "Semi-Urban", "Urban"]
}

def get_options(column):
    options = config.get("categories", {}).get(column, [])

    if not options and data is not None and column in data.columns:
        options = data[column].dropna().astype(str).unique().tolist()

    options = sorted({str(value) for value in options if pd.notna(value)})
    return options or fallback_categories[column]


prediction_tab, data_tab, performance_tab = st.tabs(
    ["ETA Prediction", "Data Visualizations", "Model Performance"]
)

# ===================== prediction_tab =====================
with prediction_tab:
    st.subheader("Enter delivery details")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            distance = st.number_input(
                "Restaurant-to-customer distance (km)",
                min_value=0.1,
                max_value=200.0,
                value=5.0,
                step=0.5
            )
            age = st.number_input(
                "Delivery person age",
                min_value=18,
                max_value=65,
                value=30,
                step=1
            )
            rating = st.slider(
                "Delivery person rating",
                min_value=1.0,
                max_value=5.0,
                value=4.5,
                step=0.1
            )
            vehicle_condition = st.selectbox(
                "Vehicle condition",
                options=[0, 1, 2],
                index=1
            )
            multiple_deliveries = st.selectbox(
                "Number of multiple deliveries",
                options=[0, 1, 2, 3],
                index=1
            )

        with right:
            order_hour = st.slider(
                "Order hour (0–23)",
                min_value=0,
                max_value=23,
                value=18
            )
            is_weekend = st.checkbox("Order is on a weekend")

            weather = st.selectbox(
                "Weather",
                get_options("Weatherconditions")
            )
            traffic = st.selectbox(
                "Traffic level",
                get_options("Road_traffic_density")
            )
            order_type = st.selectbox(
                "Order type",
                get_options("Type_of_order")
            )
            vehicle_type = st.selectbox(
                "Vehicle type",
                get_options("Type_of_vehicle")
            )
            festival = st.selectbox(
                "Festival",
                get_options("Festival")
            )
            city = st.selectbox(
                "City",
                get_options("City")
            )

        submitted = st.form_submit_button("Predict delivery time")

    if submitted:
        input_data = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Vehicle_condition": vehicle_condition,
            "multiple_deliveries": multiple_deliveries,
            "Distance_km": distance,
            "Order_Hour": order_hour,
            "Is_Weekend": int(is_weekend),
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle_type,
            "Festival": festival,
            "City": city
        }])

        predicted_minutes = predict_delivery_minutes(model, input_data)
        st.success(f"Estimated delivery time: **{predicted_minutes:.1f} minutes**")

# ====================== Data visualizations ======================
with data_tab:
    st.subheader("Dataset visualizations")

    if data is None:
        st.warning("model_ready_data.csv does not exists in data/processed/ to show charts.")
    else:
        if "Delivery_Time_min" in data.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data["Delivery_Time_min"].dropna(), bins=30, edgecolor="white")
            ax.set_title("Distribution of Delivery Time")
            ax.set_xlabel("Delivery time (minutes)")
            ax.set_ylabel("Number of deliveries")
            st.pyplot(fig)
            plt.close(fig)

        if {"Distance_km", "Delivery_Time_min"}.issubset(data.columns):
            plot_data = data[["Distance_km", "Delivery_Time_min"]].dropna()
            plot_data = plot_data.sample(
                n=min(3000, len(plot_data)),
                random_state=42
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(
                plot_data["Distance_km"],
                plot_data["Delivery_Time_min"],
                alpha=0.3
            )
            ax.set_title("Distance vs Delivery Time")
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Delivery time (minutes)")
            st.pyplot(fig)
            plt.close(fig)

        if {"Road_traffic_density", "Delivery_Time_min"}.issubset(data.columns):
            groups = []
            labels = []

            for label, group in data.groupby("Road_traffic_density"):
                values = group["Delivery_Time_min"].dropna()
                if not values.empty:
                    labels.append(str(label))
                    groups.append(values)

            if groups:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot(groups, tick_labels=labels)
                ax.set_title("Traffic Level vs Delivery Time")
                ax.set_xlabel("Traffic level")
                ax.set_ylabel("Delivery time (minutes)")
                st.pyplot(fig)
                plt.close(fig)

# ======================= Model performance =======================
with performance_tab:
    st.subheader("Final model test metrics")

    metrics = config.get("metrics", {})
    columns = st.columns(5)

    columns[0].metric(
        "MAE",
        f"{metrics['MAE']:.2f} min" if "MAE" in metrics else "N/A"
    )

    mse = metrics["RMSE"] ** 2 if "RMSE" in metrics else None
    columns[1].metric(
        "MSE",
        f"{mse:.2f}" if mse is not None else "N/A"
    )
    columns[2].metric(
        "RMSE",
        f"{metrics['RMSE']:.2f} min" if "RMSE" in metrics else "N/A"
    )
    columns[3].metric(
        "R²",
        f"{metrics['R2']:.3f}" if "R2" in metrics else "N/A"
    )
    columns[4].metric(
        "Adjusted R²",
        f"{metrics['Adjusted_R2']:.3f}" if "Adjusted_R2" in metrics else "N/A"
    )

    st.caption("MAE and RMSE are measured in minutes & R² is not classification accuracy.")

    if test_predictions is None:
        st.warning("test_predictions.csv does not exists In data/processed/ to show prediction.")
    elif {"Actual", "Predicted"}.issubset(test_predictions.columns):
        actual = test_predictions["Actual"]
        predicted = test_predictions["Predicted"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(actual, predicted, alpha=0.35)

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], "r--")

        ax.set_title("Actual vs Predicted Delivery Time")
        ax.set_xlabel("Actual time (minutes)")
        ax.set_ylabel("Predicted time (minutes)")
        st.pyplot(fig)
        plt.close(fig)

        residuals = actual - predicted

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(residuals, bins=30, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Prediction Errors")
        ax.set_xlabel("Actual minus predicted (minutes)")
        ax.set_ylabel("Number of deliveries")
        st.pyplot(fig)
        plt.close(fig)