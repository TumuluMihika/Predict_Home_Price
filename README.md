# House Price Prediction App

### **A Machine Learning + Streamlit Web Application**

This project is an end-to-end Machine Learning system that predicts **house prices** based on **house size (sq ft)**.
It includes:

* Data exploration
* Multiple ML models
* Model evaluation and comparison
* Best model selection using RMSE, R², CV-RMSE
* Ensemble learning (top-2 averaged models)
* A full interactive Streamlit web app
* Visualizations of dataset and predictions

---

# Project Structure

```
HousePricePrediction/
│── app.py                 
│── House_price.ipynb      
│── home_dataset.csv       
│── best_model.pkl         
│── README.md              
```

---

# Dataset Description

The dataset contains **home size** and **price**, suitable for a regression problem.

| Column     | Description                  |
| ---------- | ---------------------------- |
| HouseSize  | Size of house in square feet |
| HousePrice | Price of the house           |

---

# Features of This Project

### **1. Exploratory Data Analysis (EDA)**

* Scatter plot of size vs price
* Distribution plots
* Initial understanding of relationship

### **2. Machine Learning Models Implemented**

* Linear Regression
* Polynomial Regression (degree 2 & 3)
* Decision Tree Regressor
* Random Forest Regressor
* **Ensemble Model (average of best two models)**

### **3. Performance Metrics**

For each model:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score
* 7-fold Cross-validated RMSE

### **4. Best Model Selection**

Models are ranked based on:

1. RMSE
2. CV-RMSE
3. R² Score

The best model is **automatically saved to `best_model.pkl`**.

### **5. Streamlit Web App**

Built with:

```
streamlit run app.py
```

App features:

* Dataset preview
* Exploratory data visualizations
* Model training section
* Live comparison table
* Prediction with confidence interval
* Actual vs predicted plot
* Option to retrain model anytime

### **6. Confidence Interval**

Prediction is shown with a **residual-based 90th percentile interval**, giving approximate range.

---

# Installation & Setup

### Clone the repository

```
git clone https://github.com/your-username/HousePricePrediction.git
cd HousePricePrediction
```

###  Install required libraries

```
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```
pip install streamlit scikit-learn pandas numpy matplotlib joblib
```

### Run the Streamlit app

```
streamlit run app.py
```

---

# Usage Guide

### ** Predicting House Price**

1. Enter **HouseSize (sq ft)**
2. Click **Predict**
3. View:

   * Estimated price
   * Confidence interval
   * Prediction plotted on dataset

### **Model Comparison**

* Go to "Model" section
* View performance of all algorithms
* See the selected best model

### **Retrain Model**

Use Sidebar → **Retrain model (force)**
This trains all ML models again and selects the best.


# Future Improvements

* Add features like bedrooms, location, age, bathrooms
* Integrate XGBoost / LightGBM
* Apply log-transform to stabilize target distribution
* Add option to upload custom dataset
* Deploy on Streamlit Cloud / Render / HuggingFace
* Add downloadable PDF prediction report

---

# Author

**Mihika**
B.Tech CSE | ML & Data Science Enthusiast

---

