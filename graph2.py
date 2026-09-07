# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Step 2: Load dataset
base_dir = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(base_dir, "Cleaned_Car_data.csv"))

# Step 3: Prepare data
# (Drop non-useful column like name if present)
df = df.drop("name", axis=1)

# Convert categorical to numeric
df = pd.get_dummies(df)

# Step 4: Define X and y
X = df.drop("Price", axis=1)
y = df["Price"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Plot graph
plt.figure()
plt.scatter(y_test, y_pred,s=5)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()