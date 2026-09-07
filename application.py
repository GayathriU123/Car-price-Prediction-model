from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(__file__)

model_path = os.path.join(base_dir, 'LinearRegressionModel.pkl')
csv_path = os.path.join(base_dir, 'Cleaned_Car_data.csv')

model = pickle.load(open(model_path, 'rb'))
car = pd.read_csv(csv_path, index_col=0)
car.columns = car.columns.str.strip().str.lower()


# ── HOME ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


# ── PREDICT ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company   = request.form.get('company')
        car_model = request.form.get('car_models')
        year      = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven    = int(request.form.get('kilo_driven'))

        input_df = pd.DataFrame(
            [[car_model, company, year, driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )
        prediction = model.predict(input_df)
        return str(round(prediction[0], 2))
    except Exception as e:
        return str(e)


# ── INSIGHTS ────────────────────────────────────────────────
# Returns market stats for the selected car model from the CSV
@app.route('/insights', methods=['POST'])
@cross_origin()
def insights():
    try:
        company   = request.form.get('company')
        car_model = request.form.get('car_models')
        fuel_type = request.form.get('fuel_type')
        predicted = float(request.form.get('predicted_price', 0))

        # Filter by model name (and fuel type if available)
        mask = car['name'] == car_model
        subset = car[mask]

        if subset.empty:
            # Fall back to company-level data
            subset = car[car['company'] == company]

        if subset.empty:
            return jsonify({'error': 'No data found'})

        # Build year-wise average resale prices for depreciation chart
        year_avg = (
            subset.groupby('year')['price']
            .mean()
            .reset_index()
            .sort_values('year')
        )
        year_labels = year_avg['year'].astype(str).tolist()
        year_prices = [round(float(p), 0) for p in year_avg['price'].tolist()]

        # Market stats
        avg_price    = round(float(subset['price'].mean()), 0)
        min_price    = round(float(subset['price'].min()), 0)
        max_price    = round(float(subset['price'].max()), 0)
        total_listed = int(len(subset))

        # Average km driven for this model
        avg_km = round(float(subset['kms_driven'].mean()), 0) if 'kms_driven' in subset.columns else 0

        # Best year (highest resale)
        if not year_avg.empty:
            best_year = int(year_avg.loc[year_avg['price'].idxmax(), 'year'])
        else:
            best_year = None

        # How the prediction compares to market average (%)
        if avg_price > 0 and predicted > 0:
            vs_market = round(((predicted - avg_price) / avg_price) * 100, 1)
        else:
            vs_market = 0

        # Confidence score: based on how many similar listings exist
        # More listings → higher confidence (capped at 95)
        confidence = min(95, 50 + total_listed * 2)

        return jsonify({
            'avg_price':    avg_price,
            'min_price':    min_price,
            'max_price':    max_price,
            'total_listed': total_listed,
            'avg_km':       avg_km,
            'best_year':    best_year,
            'vs_market':    vs_market,
            'confidence':   confidence,
            'year_labels':  year_labels,
            'year_prices':  year_prices,
        })

    except Exception as e:
        return jsonify({'error': str(e)})


# ── COMPARE ─────────────────────────────────────────────────
# Predicts prices for two cars and returns both
@app.route('/compare', methods=['POST'])
@cross_origin()
def compare():
    try:
        results = []
        for prefix in ['a_', 'b_']:
            company   = request.form.get(prefix + 'company')
            car_model = request.form.get(prefix + 'car_models')
            year      = int(request.form.get(prefix + 'year'))
            fuel_type = request.form.get(prefix + 'fuel_type')
            driven    = int(request.form.get(prefix + 'kilo_driven'))

            input_df = pd.DataFrame(
                [[car_model, company, year, driven, fuel_type]],
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
            )
            price = round(float(model.predict(input_df)[0]), 2)

            # Get market avg for this model
            subset   = car[car['name'] == car_model]
            avg      = round(float(subset['price'].mean()), 0) if not subset.empty else 0
            listings = int(len(subset))

            results.append({
                'company':   company,
                'model':     car_model,
                'year':      year,
                'fuel':      fuel_type,
                'driven':    driven,
                'price':     price,
                'avg_price': avg,
                'listings':  listings,
            })

        return jsonify({'car_a': results[0], 'car_b': results[1]})

    except Exception as e:
        return jsonify({'error': str(e)})


# ── RUN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)

# Note: run with   python application.py