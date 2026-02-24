from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import pickle
from deep_translator import GoogleTranslator

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

# ---------------- LOAD MODEL ----------------
FAKE_MODEL_PATH = os.path.join(BASE_DIR, "ml", "fake_news_model.pkl")
FAKE_VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "fake_vectorizer.pkl")

try:
    fake_model = pickle.load(open(FAKE_MODEL_PATH, "rb"))
    fake_vectorizer = pickle.load(open(FAKE_VECTORIZER_PATH, "rb"))
    print("✅ Fake News Model Loaded Successfully")
except:
    fake_model = None
    fake_vectorizer = None
    print("⚠ Fake News Model Not Loaded")

# ---------------- HOME ----------------
@app.route("/")
def home():
    return jsonify({"message": "Backend running"})


# ---------------- FAKE NEWS DETECT ----------------
@app.route("/fake-detect", methods=["POST"])
def fake_detect():
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Translate
        english_text = GoogleTranslator(source='auto', target='en').translate(text)

        vec = fake_vectorizer.transform([english_text])

        prediction = fake_model.predict(vec)[0]
        probability = fake_model.predict_proba(vec)[0]
        confidence = round(max(probability) * 100, 2)

        result = "REAL" if prediction == 1 else "FAKE"

        # -------- SAVE HISTORY ----------
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)

        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

        history.append({
            "text": text[:120],
            "result": result,
            "confidence": confidence
        })

        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-10:], f, indent=2)

        return jsonify({
            "result": result,
            "confidence": confidence,
            "translated_text": english_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- HISTORY API ----------------
@app.route("/history", methods=["GET"])
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    return jsonify(history[::-1])


# ---------------- STATIC PAGES ----------------
@app.route("/app")
def app_page():
    return send_from_directory(PUBLIC_DIR, "index.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)