import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load dataset
base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "Cleaned_Car_data.csv"))

# ✅ Remove outliers (important for clean graph)
df = df[df["kms_driven"] < 150000]
df = df[df["Price"] < 2000000]

# Create figure
plt.figure()

# ✅ Scatter plot (clean + transparent)
plt.scatter(df["kms_driven"], df["Price"], s=8, alpha=0.4)

# ✅ Add trend line (best fit line)
z = np.polyfit(df["kms_driven"], df["Price"], 1)
p = np.poly1d(z)
plt.plot(df["kms_driven"], p(df["kms_driven"]))

# Labels and title
plt.xlabel("Kilometers Driven")
plt.ylabel("Car Price")
plt.title("Car Price vs Kilometers Driven (Cleaned Data)")

# Axis limits for better zoom
plt.xlim(0, 150000)
plt.ylim(0, 2000000)

# Grid for clarity
plt.grid()


# Show graph
plt.show()