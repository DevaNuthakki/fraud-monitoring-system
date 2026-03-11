import streamlit as st
import requests

st.set_page_config(page_title="Fraud Monitoring System", layout="wide")

st.title("Fraud Monitoring System")
st.subheader("ML Monitoring and Fraud Risk Prediction Demo")

st.write(
    """
    This system is designed to support fraud detection and ML monitoring workflows.

    It currently includes:
    - FastAPI backend for model inference
    - Streamlit frontend for user interaction
    - Real-time fraud prediction
    - Backend health check
    """
)

st.divider()

st.subheader("System Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Current Features
        - Predict fraud risk from transaction features
        - View backend connection status
        - Display fraud probability and prediction result
        """
    )

with col2:
    st.markdown(
        """
        ### Planned Features
        - Prediction logging
        - Drift monitoring
        - Alerts
        - Background monitoring jobs
        - Explainability
        """
    )

st.divider()

st.subheader("Backend Connection Check")

API_BASE = st.text_input("FastAPI Base URL", value="http://localhost:8000")

if st.button("Test Connection"):
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        response.raise_for_status()
        data = response.json()

        st.success("Backend connection successful")
        st.json(data)

    except Exception as e:
        st.error(f"Connection failed: {e}")

st.divider()

st.subheader("How to Use")

st.markdown(
    """
    1. Use the **Predict** page from the left sidebar  
    2. Enter transaction feature values  
    3. Click **Predict Fraud Risk**  
    4. View fraud probability and prediction result instantly  
    """
)