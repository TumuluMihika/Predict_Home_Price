# app.py - Streamlit app tailored to your House_price.ipynb
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------
# Config / paths
# -----------------------
DATA_PATH = "home_dataset.csv"
MODEL_PATH = "best_model.pkl"
RANDOM_STATE = 42

st.set_page_config(page_title="House Price Predictor", layout="centered")

# -----------------------
# Helper functions
# -----------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

def train_and_select_model(df):
    X = df[["HouseSize"]].values
    y = df["HousePrice"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "PolyDeg2_Linear": Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "PolyDeg3_Linear": Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)), ("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = (y_pred, model)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Cross-validated RMSE on train for insight
        try:
            cv_scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
            cv_rmse = -np.mean(cv_scores)
        except Exception:
            cv_rmse = np.nan

        results.append({"model": name, "r2": r2, "mae": mae, "rmse": rmse, "cv_rmse_train": cv_rmse})

    results_df = pd.DataFrame(results).sort_values(by=["rmse", "r2"], ascending=[True, False]).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]
    best_model = models[best_name]
    best_pred, _ = preds[best_name]

    # Save the best model to disk
    joblib.dump(best_model, MODEL_PATH)

    # Return everything useful
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results_df": results_df, "best_model_name": best_name, "best_model": best_model, "predictions": preds
    }

def model_metrics_df(results_df):
    df = results_df.copy()
    df["r2"] = df["r2"].map(lambda v: round(v, 4))
    df["mae"] = df["mae"].map(lambda v: round(v, 2))
    df["rmse"] = df["rmse"].map(lambda v: round(v, 2))
    df["cv_rmse_train"] = df["cv_rmse_train"].map(lambda v: (round(v, 2) if not pd.isna(v) else v))
    return df

def residual_confidence_interval(y_true, y_pred, percentile=90):
    res = np.abs(y_true - y_pred)
    # use percentile of residuals (e.g., 90th) as an approximate error bound
    bound = np.percentile(res, percentile)
    return bound

# -----------------------
# App: Title + load data
# -----------------------
st.title("🏠 House Price Predictor (based on HouseSize)")
st.write("This app uses the same modeling approach as your notebook. It trains / loads a model and predicts price from house size.")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Please upload `home_dataset.csv` next to this app.")
    st.stop()

df = load_data(DATA_PATH)

st.sidebar.header("Quick options")
retrain = st.sidebar.button("Retrain model (force)")

# Show dataset preview
with st.expander("📄 Dataset preview"):
    st.dataframe(df.head(50))
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -----------------------
# EDA
# -----------------------
st.header("🔎 Exploratory Data Analysis")
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Scatter: HouseSize vs HousePrice")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df["HouseSize"], df["HousePrice"], alpha=0.6)
    ax.set_xlabel("HouseSize (sqft)")
    ax.set_ylabel("HousePrice")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Distributions")
    fig2, axes = plt.subplots(2,1, figsize=(4,6))
    axes[0].hist(df["HouseSize"], bins=25)
    axes[0].set_title("HouseSize")
    axes[1].hist(df["HousePrice"], bins=25)
    axes[1].set_title("HousePrice")
    plt.tight_layout()
    st.pyplot(fig2)

# -----------------------
# Load or train model
# -----------------------
st.header("🧠 Model")
model_loaded = None
metrics_df = None
X_test = y_test = None
preds_for_plot = None
best_model_name = None

if os.path.exists(MODEL_PATH) and not retrain:
    try:
        model_loaded = load_model(MODEL_PATH)
        # We still need test data and metrics: retrain quickly (but don't overwrite model) to get those arrays
        train_info = train_and_select_model(df)  # this will overwrite file, but keeps consistent metrics
        metrics_df = train_info["results_df"]
        X_test = train_info["X_test"]
        y_test = train_info["y_test"]
        preds_for_plot = train_info["predictions"]
        best_model_name = train_info["best_model_name"]
        st.success(f"Loaded existing model `{MODEL_PATH}` and refreshed metrics.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Training a new model now...")
        train_info = train_and_select_model(df)
        metrics_df = train_info["results_df"]
        X_test = train_info["X_test"]
        y_test = train_info["y_test"]
        preds_for_plot = train_info["predictions"]
        best_model_name = train_info["best_model_name"]
        model_loaded = train_info["best_model"]
else:
    st.info("No saved model found or retrain requested — training models now (this may take a moment).")
    train_info = train_and_select_model(df)
    metrics_df = train_info["results_df"]
    X_test = train_info["X_test"]
    y_test = train_info["y_test"]
    preds_for_plot = train_info["predictions"]
    best_model_name = train_info["best_model_name"]
    model_loaded = train_info["best_model"]
    st.success(f"Training complete. Best model: {best_model_name}")

# Show metrics table
st.subheader("Model comparison")
st.dataframe(model_metrics_df(metrics_df).reset_index(drop=True))

st.write(f"**Selected model:** `{best_model_name}`")

# -----------------------
# Prediction UI
# -----------------------
st.header("📥 Predict a Price")
size_input = st.number_input("Enter house size (sqft):", min_value=100, max_value=20000, value=int(df["HouseSize"].median()), step=10)

if st.button("Predict"):
    # ensure model is loaded
    if model_loaded is None:
        st.error("Model not available.")
    else:
        # predict
        x_in = np.array([[size_input]])
        try:
            pred_price = float(model_loaded.predict(x_in)[0])
        except Exception:
            # in case model_loaded is a pipeline that expects a 1D array
            pred_price = float(model_loaded.predict(np.array([size_input]).reshape(-1,1))[0])

        # compute residual-based CI using test predictions from the selected model
        # find test predictions and calculate residual percentile
        y_pred_test = preds_for_plot[best_model_name][0] if preds_for_plot and best_model_name in preds_for_plot else None
        bound = None
        if y_pred_test is not None:
            bound = residual_confidence_interval(y_test, y_pred_test, percentile=90)
        else:
            bound = pred_price * 0.12  # fallback 12%

        lower = max(0, pred_price - bound)
        upper = pred_price + bound

        st.subheader("🔮 Prediction")
        st.write(f"**Estimated Price:** ₹ {pred_price:,.0f}")
        st.write(f"Approximate confidence interval (90th percentile residual): ₹ {lower:,.0f} — ₹ {upper:,.0f}")
        st.write(f"(Residual bound used: ₹ {bound:,.0f})")

        # Show small plot: predicted point
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.scatter(df["HouseSize"], df["HousePrice"], alpha=0.3, label="Data")
        ax3.scatter([size_input], [pred_price], color="red", s=120, label="Prediction")
        ax3.set_xlabel("HouseSize (sqft)")
        ax3.set_ylabel("HousePrice")
        ax3.set_title("Prediction on Data")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

# -----------------------
# Actual vs Predicted (for best model)
# -----------------------
st.header("📈 Actual vs Predicted (Test set)")
if preds_for_plot is not None:
    y_pred_best = preds_for_plot[best_model_name][0]
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.scatter(y_test, y_pred_best, alpha=0.6)
    mn, mx = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
    ax4.plot([mn, mx], [mn, mx], 'k--', lw=2)
    ax4.set_xlabel("Actual Price")
    ax4.set_ylabel("Predicted Price")
    ax4.set_title(f"Actual vs Predicted — {best_model_name}")
    ax4.grid(True)
    st.pyplot(fig4)

    # Show test metrics for the chosen model
    chosen_row = metrics_df[metrics_df["model"] == best_model_name].iloc[0]
    st.write("**Test metrics for chosen model:**")
    st.write({
        "r2": round(chosen_row["r2"], 4),
        "mae": round(chosen_row["mae"], 2),
        "rmse": round(chosen_row["rmse"], 2)
    })
else:
    st.info("Predictions not available to plot.")

# -----------------------
# Footer / tips
# -----------------------
st.write("---")
st.write("Tips:")
st.write("- If predictions look poor, consider adding more features (location, bedrooms, age).")
st.write("- You can click 'Retrain model (force)' in the sidebar to retrain using the dataset.")
st.write("Built from your notebook — Streamlit UI by your assistant.")
