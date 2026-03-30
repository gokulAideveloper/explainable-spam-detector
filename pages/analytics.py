import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Analytics", layout="wide")

st.title("📊 Spam Analytics Dashboard")

uploaded_file = st.file_uploader("Upload CSV for Analysis", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    if "Prediction" in df.columns:
        st.subheader("📊 Distribution")
        st.bar_chart(df["Prediction"].value_counts())

        st.subheader("📈 Probability Barchart")
        st.bar_chart(df["Spam Probability"])
    else:
        st.warning("Upload file with predictions")
