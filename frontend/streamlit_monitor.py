import os
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTIONS_LOG_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "baseline_reference.csv")
DRIFT_SUMMARY_PATH = os.path.join(BASE_DIR, "reports", "drift_summary.json")
EVIDENTLY_HTML_PATH = os.path.join(BASE_DIR, "reports", "evidently_drift_report.html")

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Drift Monitor")
st.caption("Reference vs current data drift using real prediction logs.")

# =========================
# Summary
# =========================
if os.path.exists(DRIFT_SUMMARY_PATH):
    with open(DRIFT_SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Rows", summary.get("baseline_rows", 0))
    col2.metric("Current Rows", summary.get("current_rows", 0))
    col3.metric("Columns Checked", len(summary.get("columns_checked", [])))
else:
    st.warning("Drift summary not found. Run the Evidently drift script first.")

# =========================
# Simple charts from logs
# =========================
if os.path.exists(PREDICTIONS_LOG_PATH):
    log_df = pd.read_csv(PREDICTIONS_LOG_PATH)

    st.subheader("Live Prediction Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(log_df["predicted_probability"], bins=20)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Recent Predictions")
    st.dataframe(log_df.tail(20).iloc[::-1], use_container_width=True)
else:
    st.info("No predictions log found yet.")

# =========================
# Evidently report view
# =========================
st.subheader("Evidently Drift Report")

if os.path.exists(EVIDENTLY_HTML_PATH):
    with open(EVIDENTLY_HTML_PATH, "r", encoding="utf-8") as f:
        html_report = f.read()

    components.html(html_report, height=900, scrolling=True)

    st.download_button(
        label="Download Evidently HTML Report",
        data=html_report,
        file_name="evidently_drift_report.html",
        mime="text/html"
    )
else:
    st.warning("Evidently HTML report not found. Run the drift script first.")