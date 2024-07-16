from flask import Flask, request, render_template, jsonify
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

API_KEY = 'b2261e2089259e08fa16900e6d03ec16d69dd1dae13ce12d3fef99ac7bd0017e'
URL = 'https://min-api.cryptocompare.com/data/v2/histoday'
CURRENT_URL = 'https://min-api.cryptocompare.com/data/price'
model = None


def get_crypto_data(symbol, start_date, end_date):
    try:
        start_date_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_date_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        limit = (end_date_ts - start_date_ts) // (24 * 3600)  # Calculate the number of days

        if limit > 365:
            limit = 365

        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': limit,
            'api_key': API_KEY,
            'toTs': end_date_ts,
            'e': 'CCCAGG'
        }

        response = requests.get(URL, params=params)
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Ensure the end date is included
        if df['time'].iloc[-1] != pd.to_datetime(end_date):
            additional_day_params = params.copy()
            additional_day_params['limit'] = 1
            additional_day_params['toTs'] = end_date_ts + 24 * 3600  # Add one more day to include end date
            additional_response = requests.get(URL, params=additional_day_params)
            additional_data = additional_response.json()['Data']['Data']
            additional_df = pd.DataFrame(additional_data)
            additional_df['time'] = pd.to_datetime(additional_df['time'], unit='s')
            df = pd.concat([df, additional_df])

        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def create_lagged_features(df, lag=1):
    df[f'lag_{lag}'] = df['close'].shift(lag)
    df = df.dropna()
    return df


def get_current_price(symbol):
    try:
        params = {
            'fsym': symbol,
            'tsyms': 'USD',
            'api_key': API_KEY
        }
        response = requests.get(CURRENT_URL, params=params)
        data = response.json()
        return data['USD']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    data = get_crypto_data(symbol, start_date, end_date)
    if data.empty:
        return jsonify({'error': 'Error fetching data'}), 400

    data.to_csv(f'data_{symbol}.csv')
    return jsonify({'message': 'Data fetched successfully'})


@app.route('/train_model', methods=['POST'])
def train_model_route():
    global model
    symbol = request.form['symbol']
    data_2023 = get_crypto_data(symbol, '2023-01-01', '2023-12-31')
    data_2024 = get_crypto_data(symbol, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))

    if data_2023.empty or data_2024.empty:
        return jsonify({'error': 'Error fetching data for training'}), 400

    data_2023.to_csv(f'data_{symbol}_2023.csv')
    data_2024.to_csv(f'data_{symbol}_2024.csv')

    data_2023 = create_lagged_features(data_2023, lag=1)
    data_2024 = create_lagged_features(data_2024, lag=1)

    features = ['lag_1']
    X_2023 = data_2023[features]
    y_2023 = data_2023['close']
    X_2024 = data_2024[features]
    y_2024 = data_2024['close']

    X_train, X_test, y_train, y_test = train_test_split(X_2023, y_2023, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_2023, y_2023, cv=5)
    mean_cv_score = cv_scores.mean()

    y_test_pred = model.predict(X_test)
    y_2024_pred = model.predict(X_2024)

    r2_test = r2_score(y_test, y_test_pred)
    r2_2024 = r2_score(y_2024, y_2024_pred)

    plt.figure(figsize=(14, 7))
    plt.plot(data_2023['time'], data_2023['close'], label='Actual 2023', color='blue')
    plt.plot(data_2024['time'], data_2024['close'], label='Actual 2024', color='green')
    plt.plot(data_2024['time'], y_2024_pred, label='Predicted 2024', color='red', linestyle='--')
    plt.title('Cryptocurrency Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify(
        {'cv_scores': cv_scores.tolist(), 'mean_cv_score': mean_cv_score, 'r2_test': r2_test, 'r2_2024': r2_2024,
         'plot_url': plot_url})


@app.route('/predict_price', methods=['POST'])
def predict_price():
    global model
    if model is None:
        return jsonify({'error': 'Model not trained'}), 400

    symbol = request.form['symbol']
    user_date = request.form['user_date']

    data_2023 = pd.read_csv(f'data_{symbol}_2023.csv')
    data_2024 = pd.read_csv(f'data_{symbol}_2024.csv')

    data = pd.concat([data_2023, data_2024])

    def predict_price_for_date(model, lagged_price):
        input_data = pd.DataFrame({'lag_1': [lagged_price]})
        predicted_price = model.predict(input_data)
        return predicted_price[0]

    future_date = datetime.strptime(user_date, '%Y-%m-%d')
    latest_date = pd.to_datetime(data['time']).max()
    latest_price = data['close'].iloc[-1]

    if future_date <= latest_date:
        return jsonify({'error': 'Future date must be after the latest date in the dataset'}), 400

    future_price = latest_price
    while latest_date < future_date:
        future_price = predict_price_for_date(model, future_price)
        latest_date += timedelta(days=1)

    return jsonify({'predicted_price': future_price})


if __name__ == '__main__':
    app.run(debug=True)

