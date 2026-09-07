# Car Price Predictor

A machine learning-based web application that predicts the resale price of used cars based on factors such as car model, company, year, fuel type, and kilometers driven.

The application also provides market insights and allows users to compare two cars based on their predicted prices and available market data.

## Features

- Used car price prediction
- Market price insights
- Year-wise resale price trends
- Comparison of two cars
- Average, minimum, and maximum market prices
- Average kilometers driven
- Number of available listings
- Predicted price vs. market average

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Linear Regression
- Flask
- HTML
- CSS
- JavaScript
- Bootstrap
- Chart.js

## Machine Learning

The project uses **Linear Regression** to predict used-car prices.

### Input Features

- Car model
- Company
- Year
- Kilometers driven
- Fuel type

### Workflow

1. Load the cleaned dataset
2. Prepare and preprocess the data
3. Convert categorical features into numerical values
4. Split the data into training and testing sets
5. Train the Linear Regression model
6. Generate price predictions
7. Integrate the trained model with a Flask web application

## Dataset

The project uses a cleaned used-car dataset containing information such as:

- Car name
- Company
- Year
- Price
- Kilometers driven
- Fuel type

## Application

The Flask application provides three main functions:

### 1. Price Prediction

Users can enter car details and receive an estimated resale price.

### 2. Market Insights

The application provides information such as average price, minimum price, maximum price, number of listings, average kilometers driven, and year-wise price trends.

### 3. Car Comparison

Users can enter details for two cars and compare their predicted prices and market averages.

## Project Structure

```text
carpricepredictor/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── application.py
├── Cleaned_Car_data.csv
├── LinearRegressionModel.pkl
├── graph.py
├── graph2.py
├── README.md
└── .gitignore