import streamlit as st
import pickle
import json
import hashlib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
import numpy as py
from explain import explain, generate_summary


# ==============================
# 🔐 GEMINI CONFIG (SAFE)
# ==============================
import google.generativeai as genai
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
genai.configure(api_key=os.getenv("API_KEY"))
model_ai = genai.GenerativeModel("gemini-pro")

# ==============================
# 🔐 AUTH FUNCTIONS
# ==============================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open("users.json", "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def register(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

# ==============================
# 📜 HISTORY
# ==============================

def load_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open("history.json", "w") as f:
        json.dump(history, f)

# ==============================
# 🎨 UI CONFIG
# ==============================

st.set_page_config(page_title="Spam Detector", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}
.stButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border-radius: 10px;
}
.stTextArea textarea {
    background: #1e293b;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 Explainable Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered spam detection with chatbot</div>', unsafe_allow_html=True)

# ==============================
# 🔐 LOGIN SYSTEM
# ==============================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if not st.session_state.logged_in:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if menu == "Register":
        if st.sidebar.button("Register"):
            if register(username, password):
                st.success("Registered successfully!")
            else:
                st.error("User already exists")

    if menu == "Login":
        if st.sidebar.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ==============================
# 🤖 LOAD MODEL
# ==============================

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# ==============================
# ✨ HIGHLIGHT FUNCTION
# ==============================

def highlight_text(text, explanation):
    words = text.split()
    exp_dict = dict(explanation)

    result = ""
    for word in words:
        score = exp_dict.get(word.lower(), 0)
        color = "red" if score > 0 else "green" if score < 0 else "white"
        result += f"<span style='color:{color}; font-weight:bold'>{word}</span> "
    return result

# ==============================
# 🤖 CHATBOT FUNCTION (SAFE)
# ==============================

def chatbot_response(user_msg, email, explanation, pred):
    try:
        prompt = f"""
        Email: {email}

        Prediction: {'Spam' if pred==1 else 'Not Spam'}

        Important words: {[w for w,s in explanation[:5]]}

        User question: {user_msg}

        Explain clearly in simple language.
        """

        response = model_ai.generate_content(prompt)
        return response.text

    except:
        return "⚠️ AI error. Check API key or internet."

# ==============================
# 📝 INPUT
# ==============================

st.markdown('<div class="card">', unsafe_allow_html=True)
email = st.text_area("📩 Enter Email Text")
st.markdown('</div>', unsafe_allow_html=True)

# Chat memory
if "chat" not in st.session_state:
    st.session_state.chat = []

# ==============================
# 🔍 PREDICTION
# ==============================

if st.button("Check"):

    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    st.session_state.explanation = explain(email)
    st.session_state.pred = pred

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if pred == 1:
        st.error(f"🚨 Spam ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Not Spam ({(1-prob)*100:.2f}%)")

    st.progress(int(prob * 100))
    st.write("Confidence:", prob)

    st.markdown('</div>', unsafe_allow_html=True)

    save_history({
        "text": email,
        "prediction": "Spam" if pred == 1 else "Not Spam",
        "probability": float(prob),
        "time": str(datetime.now())
    })

    explanation = st.session_state.explanation

    st.subheader("🔍 Explanation")
    st.markdown(highlight_text(email, explanation), unsafe_allow_html=True)

    st.subheader("🧠 AI Explanation")
    st.info(generate_summary(explanation))

# ==============================
# 🤖 CHATBOT UI
# ==============================

st.markdown("---")
st.subheader("🤖 AI Chat Assistant")

user_input = st.text_input("Ask about the email")

if st.button("Send"):
    if "explanation" in st.session_state:
        reply = chatbot_response(
            user_input,
            email,
            st.session_state.explanation,
            st.session_state.pred
        )
        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("Bot", reply))
    else:
        st.warning("Check email first")

for role, msg in st.session_state.chat:
    st.write(f"{'🧑' if role=='You' else '🤖'} {msg}")

# ==============================
# 📊 ANALYTICS
# ==============================

st.markdown("---")
st.subheader("📊 Analytics Dashboard")

history = load_history()

if history:
    df = pd.DataFrame(history)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(df))
    col2.metric("Spam", (df["prediction"] == "Spam").sum())
    col3.metric("Safe", (df["prediction"] == "Not Spam").sum())

    st.bar_chart(df["prediction"].value_counts())
else:
    st.info("No data yet")

# ==============================
# 📜 SIDEBAR HISTORY
# ==============================

st.sidebar.subheader("Recent Activity")

for item in history[-5:][::-1]:
    st.sidebar.write(f"{item['prediction']} ({round(item['probability']*100,1)}%)")
