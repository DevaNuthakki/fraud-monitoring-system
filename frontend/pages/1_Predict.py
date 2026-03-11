import streamlit as st
import requests

st.set_page_config(page_title="Predict Fraud Risk", layout="wide")

st.title("Predict Fraud Risk")
st.write("Enter transaction features to estimate fraud probability.")

API_BASE = st.text_input("FastAPI Base URL", value="http://localhost:8000")

st.divider()
st.subheader("Transaction Features")

col1, col2, col3 = st.columns(3)

features = []

with col1:
    time_val = st.number_input("Time", value=0.0, format="%.6f")
    features.append(time_val)

    for i in range(1, 10):
        val = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"V{i}")
        features.append(val)

with col2:
    for i in range(10, 20):
        val = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"V{i}")
        features.append(val)

with col3:
    for i in range(20, 29):
        val = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"V{i}")
        features.append(val)

    amount_val = st.number_input("Amount", value=0.0, format="%.6f")
    features.append(amount_val)

st.divider()

colA, colB, colC = st.columns(3)

with colA:
    predict_btn = st.button("Predict Fraud Risk", use_container_width=True)

with colB:
    sample_btn = st.button("Load Sample Fraud Case", use_container_width=True)

with colC:
    reset_btn = st.button("Reset Inputs", use_container_width=True)

if sample_btn:
    st.warning("Sample fraud transaction loaded. Adjust values if needed.")

if reset_btn:
    st.info("Reset the input values manually to 0.0 for now.")

if predict_btn:
    try:
        payload = {"features": features}

        response = requests.post(f"{API_BASE}/predict", json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()

        fraud_probability = result["fraud_probability"]
        threshold = result["threshold"]
        prediction = result["prediction"]

        st.divider()

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Fraud Probability", f"{fraud_probability:.4f}")

        with metric_col2:
            st.metric("Threshold", f"{threshold:.4f}")

        with metric_col3:
            st.metric("Prediction", prediction)

        if prediction == 1:
            st.error("🚨 HIGH FRAUD RISK")
        else:
            st.success("✅ LOW FRAUD RISK")

        st.subheader("API Response")
        st.json(result)

    except Exception as e:
        st.error(f"Prediction failed: {e}")