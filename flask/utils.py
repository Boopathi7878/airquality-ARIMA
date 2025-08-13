import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Get base directory (where this utils.py file is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models are stored one folder up, in "models"
PARENT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PARENT_DIR, "models")

# Static plots folder inside the Flask app
STATIC_PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)


def load_model(city):
    """Load saved ARIMA model for the given city, listing available if missing."""
    file_name = f"{city.title()}_AutoARIMA.pkl"
    model_path = os.path.join(MODELS_DIR, file_name)

    if not os.path.exists(model_path):
        available_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Available models: {available_files}"
        )
    return joblib.load(model_path)


def predict_future(city, periods):
    """Predict future AQI for the given city."""
    model = load_model(city)
    forecast = model.predict(n_periods=periods)

    start_date = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start=start_date, periods=periods)

    results = pd.DataFrame({
        "Date": future_dates,
        "Predicted_AQI": forecast
    })
    return results


def save_forecast_plot(predictions, city):
    """Save forecast plot and return relative path."""
    plt.figure(figsize=(8, 4))
    plt.plot(predictions["Date"], predictions["Predicted_AQI"], marker='o', label="Predicted AQI")
    plt.title(f"AQI Forecast for {city.title()}")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.grid(True)
    plt.legend()

    file_name = f"{city.lower()}_forecast.png"
    plot_path = os.path.join(STATIC_PLOTS_DIR, file_name)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return f"plots/{file_name}"
