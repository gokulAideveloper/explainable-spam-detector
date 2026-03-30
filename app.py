import streamlit as st
import pickle
from explain import explain
st.set_page_config(page_title="Explainable Spam Detector", layout="centered")

st.markdown("""
<style>
.big-title {font-size:36px; font-weight:700;}
.result-box {padding:10px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📧 Explainable Spam Detector</div>', unsafe_allow_html=True)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Title


# Input
email = st.text_area("Enter Email Text")


# 🔥 Highlight function
def highlight_text(text, explanation):
    words = text.split()
    exp_dict = dict(explanation)

    highlighted_text = ""

    for word in words:
        score = exp_dict.get(word.lower(), 0)

        if score > 0:
            color = "red"
        elif score < 0:
            color = "green"
        else:
            color = "black"

        highlighted_text += f'<span style="color:{color}; font-weight:bold;">{word}</span> '

    return highlighted_text


# 🔥 MAIN BUTTON LOGIC
if st.button("Check"):
    
    # 🔹 Prediction
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]


    if pred == 1:
        st.error(f"Spam 🚨 ({prob*100:.2f}%)")
    else:
        st.success(f"Not Spam ✅ ({(1-prob)*100:.2f}%)")

    st.write("Raw Probability:", prob)

    # 🔹 Explanation
    st.subheader("Explanation:")
    explanation = explain(email)

    for word, score in explanation[:5]:
        st.write(f"{word} → {score:.3f}")

    # 🔹 Highlight
    highlighted = highlight_text(email, explanation)
    st.markdown(highlighted, unsafe_allow_html=True)

    # 🔹 Top words
    st.subheader("Top Important Words 🔍")
    st.subheader("📊 Word Impact Graph")
    for word, score in explanation[:10]:
        if score > 0:
            st.write(f"🔴 {word} → Spam signal ({score:.3f})")
        else:
            st.write(f"🟢 {word} → Safe signal ({score:.3f})")

    # 🔥 GRAPH (MUST BE INSIDE BUTTON)
    import matplotlib.pyplot as plt

    top = sorted(explanation[:10], key=lambda x: x[1])
    words = [w for w, s in top]
    scores = [s for w, s in top]
    colors = ["red" if s > 0 else "green" for s in scores]
    plt.figure()
    plt.barh(words, scores, color=colors)
    plt.xlabel("Importance Score")
    plt.title("Top Words Impact")

    st.pyplot(plt)