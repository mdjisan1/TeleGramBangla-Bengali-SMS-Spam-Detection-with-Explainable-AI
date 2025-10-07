from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from lime.lime_text import LimeTextExplainer

# --- Load saved model & vectorizer ---
with open("best_spam_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# --- Flask app ---
app = Flask(__name__)

# --- Text cleaning function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\+?\d{10,13}", "<PHONE>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Safe probability prediction ---
def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        # Convert decision function to pseudo-probability
        decision = model.decision_function(X)
        probs = 1 / (1 + np.exp(-decision))  # Sigmoid
        # Ensure shape (n_samples, 2)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        return probs
    else:
        raise AttributeError("Model has neither predict_proba nor decision_function")

# --- Prediction with confidence ---
def predict_sms(message):
    cleaned = clean_text(message)
    features = tfidf.transform([cleaned])
    probs = safe_predict_proba(best_model, features)
    prediction = np.argmax(probs)
    confidence = probs[0][prediction]
    label = "Spam" if prediction == 1 else "Ham"
    return label, confidence

# --- LIME explanation ---
def explain_prediction(message):
    explainer = LimeTextExplainer(class_names=["Ham", "Spam"])

    def predict_proba_fn(texts):
        cleaned_texts = [clean_text(t) for t in texts]
        features = tfidf.transform(cleaned_texts)
        return safe_predict_proba(best_model, features)

    explanation = explainer.explain_instance(message, predict_proba_fn, num_features=5)
    explanation_list = explanation.as_list()

    # Map token fragments back to actual words
    original_words = message.split()
    mapped_words = []
    for token, weight in explanation_list:
        match = next((word for word in original_words if token in word), token)
        mapped_words.append((match, weight))

    weights = [abs(weight) for _, weight in mapped_words]
    total = sum(weights) if sum(weights) > 0 else 1
    top_words = [(word, round(abs(weight) / total * 100, 1)) for word, weight in mapped_words]

    return top_words[:5]

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    top_words = []
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()
        if message_text:
            result, confidence = predict_sms(message_text)
            confidence = round(confidence * 100, 2)
            top_words = explain_prediction(message_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message_text=message_text,
        top_words=top_words
    )

if __name__ == "__main__":
    app.run(debug=True)
