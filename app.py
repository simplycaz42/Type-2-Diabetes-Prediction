# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ---------------------------
# Page config & simple theme
# ---------------------------
st.set_page_config(
    page_title="Type 2 Diabetes Risk Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS (subtle, professional)
st.markdown("""
<style>
/* Page background and cards */
.reportview-container .main .block-container {padding-top: 1rem; padding-bottom: 2rem;}
/* Buttons */
.stButton>button {border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600;}
/* Metric boxes */
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
/* Tables */
thead tr th {text-align: left !important;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load("diabetes_model.pkl")   # RandomForest (recommended) or any clf with predict_proba
    scaler = joblib.load("scaler.pkl")          # StandardScaler fitted on training data
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Could not load model/scaler: {e}")
    st.stop()

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

# Healthy ranges for simple benchmarking (informational)
HEALTHY_INFO = {
    "Glucose": "70–130 mg/dL (fasting)",
    "BloodPressure": "<120/80 mmHg (normal)",
    "BMI": "18.5–24.9",
    "SkinThickness": "≈ 7–50 mm (triceps skinfold typical lab range)",
    "Insulin": "≈ 16–166 μU/mL (fasting; lab-dependent)",
    "DiabetesPedigreeFunction": "Relative risk proxy (0.0–~2.5 typical)",
    "Pregnancies": "0–10 (typical observed in dataset)",
    "Age": "0–120 years"
}

# Midpoints to visualize “healthy target” (rough, for comparison chart)
HEALTHY_MIDPOINTS = {
    "Glucose": 100,              # middle of 70–130
    "BloodPressure": 80,         # diastolic-ish single value (dataset uses single BP, treat as 80 ref)
    "BMI": 22,                   # mid of 18.5–24.9
    "SkinThickness": 25,         # rough mid
    "Insulin": 90,               # rough mid within lab range
    "DiabetesPedigreeFunction": 0.5,
    "Pregnancies": 0,
    "Age": 30
}

# ---------------------------
# Helper functions
# ---------------------------
def risk_category(prob: float) -> str:
    """Return Low/Moderate/High based on probability [0..1]."""
    if prob < 0.33: return "Low"
    if prob < 0.66: return "Moderate"
    return "High"

def make_explanation(values_dict):
    """Very simple rule-based explanation using common sense thresholds."""
    notes = []
    if values_dict["Glucose"] > 130:
        notes.append("Glucose above normal fasting range.")
    elif values_dict["Glucose"] >= 110:
        notes.append("Glucose slightly elevated (pre-diabetic range possible).")

    if values_dict["BMI"] >= 30:
        notes.append("BMI in the obese range.")
    elif values_dict["BMI"] >= 25:
        notes.append("BMI in the overweight range.")

    if values_dict["BloodPressure"] > 120:
        notes.append("Blood pressure above normal (<120 ideal).")

    if values_dict["Insulin"] > 166:
        notes.append("Insulin higher than typical fasting range.")

    if values_dict["Age"] >= 45:
        notes.append("Age is a known risk factor (≥45).")

    if values_dict["DiabetesPedigreeFunction"] >= 1.0:
        notes.append("Family history indicator (DPF) is relatively high.")

    if not notes:
        notes.append("Most entered values are within commonly referenced ranges.")
    return notes

def feature_importances_df(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
    return None

def to_history_row(values_dict, prob, pred, notes):
    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **values_dict,
        "Predicted_Class": int(pred),
        "Risk_Probability_%": round(prob * 100, 2),
        "Risk_Category": risk_category(prob),
        "Notes": " | ".join(notes)
    }

# ---------------------------
# Sidebar (About + Ranges)
# ---------------------------
st.sidebar.title("ℹ️ About this App")
st.sidebar.write(
    "Predict **Type 2 Diabetes risk** using a ML model trained on the **PIMA Indians Diabetes** dataset. "
    "Enter patient metrics, then view the predicted risk, explanations, and charts."
)

st.sidebar.subheader("📘 Healthy Reference Ranges")
for k, v in HEALTHY_INFO.items():
    st.sidebar.caption(f"**{k}**: {v}")

st.sidebar.markdown("---")
st.sidebar.caption("This tool is for **educational purposes** and not a substitute for professional medical advice.")

# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([1, 1.4])

with left:
    st.header("Patient Inputs")

    # All start at 0 by request (sliders where nice)
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Number of times pregnant.")
    glucose = st.slider("Glucose (mg/dL)", min_value=0, max_value=300, value=0, step=1, help="Fasting plasma glucose.")
    blood_pressure = st.slider("Blood Pressure (single value)", min_value=0, max_value=200, value=0, step=1, help="Dataset uses a single BP value; <120 is considered normal.")
    skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=0, step=1, help="Triceps skinfold thickness; 0 often means missing in dataset.")
    insulin = st.slider("Insulin (μU/mL)", min_value=0, max_value=900, value=0, step=1, help="Fasting insulin; 0 can represent missing in dataset.")
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1, help="Body Mass Index.")
    dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.0, step=0.01, help="Proxy for family history of diabetes.")
    age = st.slider("Age (years)", min_value=0, max_value=120, value=0, step=1, help="Patient age.")

    predict_clicked = st.button("🔮 Predict Risk", use_container_width=True)

with right:
    st.header("Results & Visualizations")
    # Reserve containers we can update after click
    metric_cols = st.columns(3)
    gauge_placeholder = st.empty()
    explain_placeholder = st.empty()
    charts_tab, fi_tab, history_tab = st.tabs(["📊 Comparisons", "⭐ Feature Importance", "🕘 History"])

# ---------------------------
# Prediction flow
# ---------------------------
if predict_clicked:
    # Prepare input
    X_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, dpf, age]], dtype=float)

    # Scale if scaler provided
    try:
        X_scaled = scaler.transform(X_input)
    except Exception:
        X_scaled = X_input  # Fallback if scaler not compatible

    # Predict and proba
    try:
        y_proba = model.predict_proba(X_scaled)[0][1]  # probability of class 1
    except Exception:
        # If no predict_proba, approximate with decision_function or 0/1
        try:
            y_raw = model.decision_function(X_scaled)[0]
            # Simple logistic squash for demo; your model might differ
            y_proba = 1 / (1 + np.exp(-y_raw))
        except Exception:
            y_pred = int(model.predict(X_scaled)[0])
            y_proba = float(y_pred)

    y_pred = int(model.predict(X_scaled)[0])
    cat = risk_category(y_proba)

    # Top metrics
    with right:
        with metric_cols[0]:
            st.metric("Risk Category", cat)
        with metric_cols[1]:
            st.metric("Risk Probability", f"{y_proba*100:.2f} %")
        with metric_cols[2]:
            st.metric("Predicted Class", y_pred)

        # Color-coded banner
        if cat == "Low":
            st.success("Low Risk predicted.")
        elif cat == "Moderate":
            st.warning("Moderate Risk predicted.")
        else:
            st.error("High Risk predicted.")

        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(y_proba*100),
            number={'suffix': "%"},
            title={'text': "Diabetes Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.3},
                'steps': [
                    {'range': [0, 33], 'color': "#90EE90"},
                    {'range': [33, 66], 'color': "#FFD580"},
                    {'range': [66, 100], 'color': "#FF9E9E"}
                ]
            }
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        gauge_placeholder.plotly_chart(gauge, use_container_width=True)

        # Explanation bullets
        values_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }
        notes = make_explanation(values_dict)
        explain_placeholder.markdown("### 🧠 Why this result?")
        for n in notes:
            st.markdown(f"- {n}")

        # Tabs
        with charts_tab:
            # Comparison bar chart (user vs healthy midpoint)
            comp_df = pd.DataFrame({
                "Feature": FEATURES,
                "Your Value": [values_dict[f] for f in FEATURES],
                "Healthy Reference": [HEALTHY_MIDPOINTS[f] for f in FEATURES]
            })
            comp_long = comp_df.melt(id_vars="Feature", var_name="Type", value_name="Value")
            comp_fig = px.bar(comp_long, x="Feature", y="Value", color="Type", barmode="group")
            comp_fig.update_layout(height=420, xaxis_tickangle=-30, margin=dict(l=10, r=10, t=30, b=120))
            st.plotly_chart(comp_fig, use_container_width=True)

        with fi_tab:
            fi_df = feature_importances_df(model, FEATURES)
            if fi_df is not None:
                st.caption("Feature importance from the trained model (higher = more influence).")
                fi_fig = px.bar(fi_df, x="Feature", y="Importance")
                fi_fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=60))
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info("Your model does not expose `feature_importances_` (e.g., Logistic Regression).")

        # History (session) + downloads
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(to_history_row(values_dict, y_proba, y_pred, notes))

        with history_tab:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True, height=260)

            # Download buttons
            st.download_button(
                "⬇️ Download This Result (CSV)",
                pd.DataFrame([hist_df.iloc[-1]]).to_csv(index=False).encode("utf-8"),
                file_name="diabetes_prediction_result.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "⬇️ Download Full Session History (CSV)",
                hist_df.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("© 2025 Diabetes Risk Dashboard • Educational use only.")
