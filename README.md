# 🚗 Car Price Prediction with Machine Learning

A beginner-friendly Machine Learning project that predicts the **selling price of a used car** based on features like brand, fuel type, mileage, transmission, and car age.

---

## 📌 Project Overview

The price of a used car depends on many factors — brand reputation, fuel type, kilometers driven, age of the car, and more. This project builds a **Random Forest Regression model** to predict car selling prices with high accuracy.

**Model Accuracy: ~95.8% (R² Score)**

---

## 📁 Project Structure

```
car_price_prediction/
│
├── car_price_prediction.py   # Main Python script (full pipeline)
├── car_data.csv              # Dataset (301 cars)
├── eda_plots.png             # EDA visualizations
├── actual_vs_predicted.png   # Model performance plot
├── feature_importance.png    # Feature importance chart
└── README.md                 # Project documentation
```

---

## 📊 Dataset

| Column | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Year of purchase |
| Selling_Price | Price the owner wants to sell at (Target) |
| Present_Price | Current ex-showroom price |
| Driven_kms | Total kilometers driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Selling_type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

---

## ⚙️ Steps in the Project

1. **Import Libraries** — pandas, numpy, sklearn, matplotlib, seaborn
2. **Load Dataset** — 301 rows, 9 columns
3. **Exploratory Data Analysis (EDA)** — distributions, correlation heatmap
4. **Feature Engineering** — created `Car_Age` from `Year`
5. **Data Preprocessing** — label encoding of categorical columns
6. **Train/Test Split** — 80% train, 20% test
7. **Model Training** — Random Forest Regressor (100 trees)
8. **Model Evaluation** — R² Score, Mean Absolute Error
9. **Visualization** — Actual vs Predicted, Feature Importance
10. **Prediction** — predict price for a new car input

---

## 📈 Results

| Metric | Value |
|---|---|
| R² Score | **0.9583 (95.83%)** |
| Mean Absolute Error | **₹ 0.65 Lakhs** |

---

## 🧰 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/car_price_prediction.git
cd car_price_prediction

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 3. Run the script
python car_price_prediction.py
```

---

## 🙋 Author

**Sk Sahil**  
B.Tech CSE | GCET  
GitHub: [shaiksahil-123](https://github.com/shaiksahil-123)
