"""
Modelling the Tensile and Corrosion Properties of High Entropy Alloys
---------------------------------------------------------------------
This script simulates data and provides a full machine learning pipeline with an interactive Streamlit app.
It allows users to vary elemental compositions and observe predicted material properties.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# --------------------------
# 1. Generate Synthetic Dataset
# --------------------------
np.random.seed(42)

# Generate 500 alloy compositions with random wt% for 5 elements
n_samples = 500
data = {
    "Al (wt%)": np.random.uniform(5, 30, n_samples),
    "Fe (wt%)": np.random.uniform(5, 30, n_samples),
    "Cu (wt%)": np.random.uniform(5, 30, n_samples),
    "Mn (wt%)": np.random.uniform(5, 30, n_samples),
    "Cr (wt%)": np.random.uniform(5, 30, n_samples),
}

df = pd.DataFrame(data)

# Normalize wt% to total 100%
df_total = df.sum(axis=1)
df = df.div(df_total, axis=0) * 100


# Simulate target variables
def simulate_tensile(row):
    return 250 + row["Fe (wt%)"] * 2 + row["Cr (wt%)"] * 1.5 - row["Cu (wt%)"] * 1.2 + np.random.normal(0, 10)


def simulate_corrosion(row):
    return 6.0 - row["Cr (wt%)"] * 0.15 + row["Cu (wt%)"] * 0.1 + np.random.normal(0, 0.3)


df["Tensile Strength (MPa)"] = df.apply(simulate_tensile, axis=1)
df["Corrosion Rate (mm/year)"] = df.apply(simulate_corrosion, axis=1)


# --------------------------
# 2. Exploratory Data Analysis
# --------------------------
def run_eda():
    st.subheader("Exploratory Data Analysis")
    st.write(df.describe())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# --------------------------
# 3. Model Training
# --------------------------
features = ["Al (wt%)", "Fe (wt%)", "Cu (wt%)", "Mn (wt%)", "Cr (wt%)"]
X = df[features]
y_tensile = df["Tensile Strength (MPa)"]
y_corrosion = df["Corrosion Rate (mm/year)"]

X_train, X_test, y_t_train, y_t_test = train_test_split(X, y_tensile, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_c_train, y_c_test = train_test_split(X, y_corrosion, test_size=0.2, random_state=42)

model_tensile = RandomForestRegressor(n_estimators=200, random_state=42)
model_corrosion = RandomForestRegressor(n_estimators=200, random_state=42)

model_tensile.fit(X_train, y_t_train)
model_corrosion.fit(X_train_c, y_c_train)


# --------------------------
# 4. Evaluation
# --------------------------
def model_eval():
    st.subheader("Model Evaluation")
    pred_t = model_tensile.predict(X_test)
    pred_c = model_corrosion.predict(X_test_c)

    st.write("**Tensile Strength Model**")
    st.write("R²:", r2_score(y_t_test, pred_t))
    st.write("MSE:", mean_squared_error(y_t_test, pred_t))

    st.write("**Corrosion Rate Model**")
    st.write("R²:", r2_score(y_c_test, pred_c))
    st.write("MSE:", mean_squared_error(y_c_test, pred_c))


# --------------------------
# 5. Interactive Prediction
# --------------------------
def interactive_prediction():
    st.subheader("Interactive Alloy Property Predictor")
    st.write("Adjust the composition of each element (total will be auto-normalized).")

    al = st.slider("Al (wt%)", 0.0, 100.0, 20.0)
    fe = st.slider("Fe (wt%)", 0.0, 100.0, 20.0)
    cu = st.slider("Cu (wt%)", 0.0, 100.0, 20.0)
    mn = st.slider("Mn (wt%)", 0.0, 100.0, 20.0)
    cr = st.slider("Cr (wt%)", 0.0, 100.0, 20.0)

    comp = pd.DataFrame([[al, fe, cu, mn, cr]], columns=features)
    comp = comp.div(comp.sum(axis=1)[0]) * 100  # Normalize to 100%

    pred_t = model_tensile.predict(comp)[0]
    pred_c = model_corrosion.predict(comp)[0]

    st.success(f"Predicted Tensile Strength: **{pred_t:.2f} MPa**")
    st.warning(f"Predicted Corrosion Rate: **{pred_c:.2f} mm/year**")


# --------------------------
# 6. Streamlit App Layout
# --------------------------
def main():
    st.title("High Entropy Alloy Modeller")
    
    # === HEADER with emojis and quotes ===
    st.markdown(
        """
        <div style="text-align: center; font-weight: 700; font-size: 24px; color: #FF4500; margin-bottom: 5px;">
            🚀🔥 Developed by <span style="color:#00CED1;">PRAISE ADEYEYE</span> 🔥🚀
        </div>
        <div style="text-align: center; font-style: italic; font-size: 18px; color: #32CD32; margin-bottom: 20px;">
            "Engineering the future, one alloy at a time!" 💥⚙️💡
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### Tensile and Corrosion Property Prediction for High Entropy Alloys")

    menu = ["Prediction Tool", "Data Summary", "Model Evaluation"]
    choice = st.sidebar.selectbox("Select Activity", menu)

    if choice == "Prediction Tool":
        interactive_prediction()
    elif choice == "Data Summary":
        run_eda()
    elif choice == "Model Evaluation":
        model_eval()
        
    # === FOOTER with copyright info ===
    st.markdown(
        """
        <hr style="margin-top: 50px; margin-bottom: 10px;">
        <div style="text-align: center; font-size: 14px; color: #888888;">
            &copy; {year} Praise Adeyeye. All rights reserved.
        </div>
        """.format(year=pd.Timestamp.now().year),
        unsafe_allow_html=True,
    )
