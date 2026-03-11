import os
import json
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALERTS_PATH = os.path.join(BASE_DIR, "reports", "alerts.json")

st.set_page_config(page_title="Alerts", layout="wide")

st.title("Alerts")
st.caption("Model monitoring alerts based on drift thresholds.")

if not os.path.exists(ALERTS_PATH):
    st.warning("alerts.json not found. Run the drift monitoring script first.")
    st.stop()

with open(ALERTS_PATH, "r") as f:
    alerts = json.load(f)

if len(alerts) == 0:
    st.info("No alerts generated yet.")
    st.stop()

alerts_df = pd.DataFrame(alerts)

open_alerts = alerts_df[alerts_df["status"] == "open"]
closed_alerts = alerts_df[alerts_df["status"] == "closed"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Alerts", len(alerts_df))
col2.metric("Open Alerts", len(open_alerts))
col3.metric("Closed Alerts", len(closed_alerts))

st.subheader("Open Alerts")

if not open_alerts.empty:
    st.dataframe(open_alerts, use_container_width=True)
else:
    st.info("No open alerts.")

st.subheader("Closed Alerts")

if not closed_alerts.empty:
    st.dataframe(closed_alerts, use_container_width=True)
else:
    st.info("No closed alerts.")

st.subheader("Alert Details")

selected_alert = st.selectbox("Select Alert ID", alerts_df["id"])

alert_data = alerts_df[alerts_df["id"] == selected_alert].iloc[0]

st.json(alert_data.to_dict())