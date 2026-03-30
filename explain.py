import pickle
import shap

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


# 🔥 WORD-LEVEL EXPLANATION (for highlight)
def explain(text):
    # Convert text to vector
    vec = vectorizer.transform([text])

    # Get model coefficients
    coefs = model.coef_[0]

    # Get non-zero indices (important words only)
    indices = vec.indices

    # Calculate scores
    scores = vec.data * coefs[indices]

    # Get corresponding words
    words = vectorizer.get_feature_names_out()[indices]

    # Combine words + scores
    word_scores = list(zip(words, scores))

    # Sort by importance
    return sorted(word_scores, key=lambda x: x[1], reverse=True)


def shap_explain(text):
    import shap
    
    # Convert text
    vec = vectorizer.transform([text])
    vec_dense = vec.toarray()
    
    # 🔥 Use predict_proba (THIS IS KEY FIX)
    explainer = shap.Explainer(model.predict_proba, vec_dense)
    
    shap_values = explainer(vec_dense)
    
    return shap_values
def generate_summary(explanation):
    spam_words = [w for w, s in explanation if s > 0][:3]
    safe_words = [w for w, s in explanation if s < 0][:3]

    if spam_words:
        return f"This email looks like SPAM because of words like {', '.join(spam_words)}."
    elif safe_words:
        return f"This email looks SAFE due to words like {', '.join(safe_words)}."
    else:
        return "No strong indicators found."
