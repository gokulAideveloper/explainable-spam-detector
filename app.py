import streamlit as st
import pickle
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
from explain import explain

# ==============================
# 🔐 AUTH FUNCTIONS (TOP)
# ==============================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open("users.json", "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                return {}  # fix corrupted file
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
    if username in users and users[username] == hash_password(password):
        return True
    return False

# ==============================
# 📜 HISTORY FUNCTIONS
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
# 🎨 UI SETTINGS
# ==============================

st.set_page_config(page_title="Spam Detector", layout="centered")

st.markdown("""
<style>
body {background-color:#0f172a; color:white;}
.stTextArea textarea {
    background:#1e293b; color:white; border-radius:10px;
}
.stButton button {
    background:#6366f1; color:white; border-radius:10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style='text-align:center; font-size:42px; color:#38bdf8;'>
🚀 Explainable Spam Detector
</h1>
<p style='text-align:center; color:gray;'>
AI-powered email analysis with explainability
</p>
""", unsafe_allow_html=True)
st.title("📧 Explainable Spam Detector")
st.markdown('<div style="padding:20px; border-radius:15px; background:#1e293b;">', unsafe_allow_html=True)

# your result here

st.markdown('</div>', unsafe_allow_html=True)

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
                st.sidebar.success("Registered successfully!")
            else:
                st.sidebar.error("User already exists")

    if menu == "Login":
        if st.sidebar.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")

    st.stop()
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
    st.rerun()

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

        if score > 0:
            color = "red"
        elif score < 0:
            color = "green"
        else:
            color = "white"

        result += f"<span style='color:{color}; font-weight:bold'>{word}</span> "

    return result

# ==============================
# 📝 INPUT
# ==============================

email = st.text_area("Enter Email Text")

# ==============================
# 🔍 PREDICTION
# ==============================

if st.button("Check"):

    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    st.progress(int(prob * 100))

    if pred == 1:
        st.error(f"Spam 🚨 ({prob*100:.2f}%)")
    else:
        st.success(f"Not Spam ✅ ({(1-prob)*100:.2f}%)")

    st.write("Confidence:", prob)

    # Save history
    save_history({
        "text": email,
        "prediction": "Spam" if pred == 1 else "Not Spam",
        "probability": float(prob),
        "time": str(datetime.now())
    })
 # =================================
    #Bottom
# ===================================
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

history = load_history()

if len(history) > 0:
    total = len(history)
    spam = sum(1 for h in history if h["prediction"] == "Spam")
    not_spam = total - spam

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Checked", total)
    col2.metric("Spam Emails", spam)
    col3.metric("Safe Emails", not_spam)

    import pandas as pd
    df = pd.DataFrame(history)

    st.bar_chart(df["prediction"].value_counts())
else:
    st.info("No data yet")
    # ==============================
    # 📊 EXPLANATION
    # ==============================

    explanation = explain(email)

    st.subheader("Explanation")
    for word, score in explanation[:5]:
        st.write(f"{word} → {score:.3f}")

    st.markdown(highlight_text(email, explanation), unsafe_allow_html=True)

    # ==============================
    # 📈 GRAPH
    # ==============================

    st.subheader("📊 Word Impact Graph")

    top = sorted(explanation[:10], key=lambda x: x[1])
    words = [w for w, s in top]
    scores = [s for w, s in top]
    colors = ["red" if s > 0 else "green" for s in scores]

    plt.figure()
    plt.barh(words, scores, color=colors)
    plt.xlabel("Impact")
    plt.title("Top Words")

    st.pyplot(plt)

# ==============================
# 📜 HISTORY VIEW
# ==============================

st.sidebar.subheader("History")

history = load_history()

for item in history[-5:]:
    st.sidebar.write(f"{item['prediction']} - {item['time']}")
st.sidebar.subheader("Recent Activity")

for item in history[-5:][::-1]:
    st.sidebar.write(f"{item['prediction']} ({round(item['probability']*100,1)}%)")
