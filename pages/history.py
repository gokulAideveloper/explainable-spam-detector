import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="History", layout="wide")

st.title("🗂 Prediction History")

# Load data
def load_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except:
        return []

data = load_history()
user = st.session_state.get("user", None)

data = load_history()

# Filter only current user
data = [d for d in data if d.get("user") == user]
if len(data) == 0:
    st.warning("No history available")
else:
    df = pd.DataFrame(data)

    st.subheader("📊 All Predictions")
    st.dataframe(df)

    # Download
    st.download_button(
        "⬇ Download History",
        df.to_csv(index=False),
        "history.csv"
    )