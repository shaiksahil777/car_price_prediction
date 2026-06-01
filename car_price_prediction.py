
#   CAR PRICE PREDICTION WITH MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("   CAR PRICE PREDICTION — Machine Learning Project")
print("=" * 55)

df = pd.read_csv("car_data.csv")

print("\n📦 Dataset loaded successfully!")
print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

# ──  BASIC DATA EXPLORATION ───────
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🔍 Missing Values:")
print(df.isnull().sum())

# ──  DATA PREPROCESSING ─────────
df["Car_Age"] = 2024 - df["Year"]
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

df["Fuel_Type"]     = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Selling_type"]  = df["Selling_type"].map({"Dealer": 0, "Individual": 1})
df["Transmission"]  = df["Transmission"].map({"Manual": 0, "Automatic": 1})

print("\n✅ Preprocessing done!")
print("Encoded dataset sample:")
print(df.head())

# ── VISUALIZATIONS ───────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Car Price Prediction — Data Insights", fontsize=15, fontweight="bold")

# Plot 1: Selling Price Distribution
axes[0, 0].hist(df["Selling_Price"], bins=30, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Distribution of Selling Price")
axes[0, 0].set_xlabel("Selling Price (Lakhs)")
axes[0, 0].set_ylabel("Count")

# Plot 2: Selling Price vs Present Price
axes[0, 1].scatter(df["Present_Price"], df["Selling_Price"],
                   alpha=0.6, color="coral", edgecolors="white", linewidths=0.5)
axes[0, 1].set_title("Present Price vs Selling Price")
axes[0, 1].set_xlabel("Present Price (Lakhs)")
axes[0, 1].set_ylabel("Selling Price (Lakhs)")

# Plot 3: Fuel Type count
fuel_labels = ["Petrol", "Diesel", "CNG"]
fuel_counts = df["Fuel_Type"].value_counts().sort_index()
axes[1, 0].bar(fuel_labels[:len(fuel_counts)], fuel_counts.values,
               color=["#4CAF50", "#2196F3", "#FF9800"])
axes[1, 0].set_title("Cars by Fuel Type")
axes[1, 0].set_xlabel("Fuel Type")
axes[1, 0].set_ylabel("Count")

# Plot 4: Correlation Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1, 1].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 EDA plots saved → eda_plots.png")

# ── FEATURE & TARGET SPLIT ───────────
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print(f"\n🎯 Features: {list(X.columns)}")
print(f"   Target  : Selling_Price")

# ── TRAIN / TEST SPLIT ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n🔀 Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── TRAIN MODEL ───────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n🤖 Random Forest model trained!")

# ── EVALUATE MODEL ─────────────
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 45)
print("   MODEL PERFORMANCE")
print("=" * 45)
print(f"   R² Score (Accuracy)  : {r2:.4f}  ({r2*100:.2f}%)")
print(f"   Mean Absolute Error  : ₹ {mae:.4f} Lakhs")
print("=" * 45)

# ── ACTUAL vs PREDICTED PLOT ──────
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="royalblue", edgecolors="white", linewidths=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title(f"Actual vs Predicted Price  |  R² = {r2:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Actual vs Predicted plot saved → actual_vs_predicted.png")

# ── FEATURE IMPORTANCE ───────
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="teal", edgecolor="white")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Feature importance plot saved → feature_importance.png")

# ──  PREDICT ON NEW DATA ──────
print("\n🚗 Sample Prediction:")
sample = pd.DataFrame({
    "Present_Price": [6.0],
    "Driven_kms":    [40000],
    "Fuel_Type":     [0],        
    "Selling_type":  [0],        
    "Transmission":  [0],      
    "Owner":         [0],
    "Car_Age":       [5]
})

predicted_price = model.predict(sample)[0]
print(f"   Input  → Present Price: ₹6L | Driven: 40,000 km | Age: 5 yrs | Petrol | Manual")
print(f"   Output → Predicted Selling Price: ₹ {predicted_price:.2f} Lakhs")

print("\n✅ Project complete! All outputs saved.")
