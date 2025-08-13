import os
import zipfile
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /mount/src/airquality-arima/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # /mount/src/airquality-arima
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")       # /mount/src/airquality-arima/models
ZIP_PATH = os.path.join(PROJECT_ROOT, "models.zip")     # /mount/src/airquality-arima/models.zip

# --- Unzip models if missing or empty ---
if os.path.exists(ZIP_PATH) and (not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR)):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODELS_DIR)
    print(f"✅ Extracted models.zip to {MODELS_DIR}")
    print(f"📂 Extracted files: {os.listdir(MODELS_DIR)}")
else:
    print("ℹ️ Models folder already populated or models.zip not found.")

# --- Utility Functions ---
def list_available_cities():
    """Scan models folder and return city names from *_AutoARIMA.pkl files."""
    if not os.path.exists(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_AutoARIMA.pkl")]
    cities = [f.replace("_AutoARIMA.pkl", "") for f in files]
    return sorted(cities)

def load_model(city):
    """Load the saved ARIMA model for the selected city."""
    file_name = f"{city}_AutoARIMA.pkl"
    model_path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def predict_future(city, periods):
    """Predict future AQI values for given city and periods from today."""
    model = load_model(city)
    forecast = model.predict(n_periods=periods)

    start_date = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start=start_date, periods=periods)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_AQI": forecast
    })

# --- Streamlit UI ---
st.title("🌆 AQI Forecasting with ARIMA")

available_cities = list_available_cities()

if not available_cities:
    st.warning(f"No model files found in '{MODELS_DIR}'. Please add *_AutoARIMA.pkl files.")
else:
    city_name = st.selectbox("Select city:", available_cities)
    days = st.number_input("Number of days to forecast:", min_value=1, max_value=60, value=7)

    if st.button("Generate Forecast"):
        try:
            output_df = predict_future(city_name, days)
            st.subheader(f"Predicted AQI for {city_name} (Next {days} Days)")

            # Display table
            st.dataframe(output_df.style.format({"Predicted_AQI": "{:.2f}"}))

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(output_df["Date"], output_df["Predicted_AQI"], marker='o', linestyle='-', color='b')
            ax.set_title(f"Predicted AQI for {city_name} (Next {days} Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("AQI")
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
