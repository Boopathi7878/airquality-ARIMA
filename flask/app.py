import os
import pandas as pd
from flask import Flask, render_template, request
from utils import predict_future, save_forecast_plot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['city'].strip()
        days = int(request.form['days'])

        predictions_df = predict_future(city, days)
        plot_path = save_forecast_plot(predictions_df, city)

        return render_template(
            'result.html',
            city=city,
            predictions=predictions_df.to_dict(orient='records'),
            plot_path=plot_path
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
