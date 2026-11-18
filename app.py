
import pickle
import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Model file paths
model_path = BASE_DIR / "svm_model.pkl"
vectorizer_path = BASE_DIR / "tfidf.pkl"
scaler_path = BASE_DIR / "scaler.pkl"

# Load the trained model and vectorizer
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

print("Model loaded successfully!")
print("Vocabulary Size:", len(vectorizer.vocabulary_))


# Clean text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# Preprocess text
def preprocess_text(text):
    text = clean_text(text)

    features = vectorizer.transform([text])

    # Convert sparse matrix into dense array if needed
    if hasattr(features, "toarray"):
        features = features.toarray()

    return features


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get input text
        data = request.get_json(silent=True) or {}
        text = data.get('text', '')

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Please enter some text"}), 400

        # Handle encoding
        text = text.encode('utf-8', 'ignore').decode('utf-8')

        # Preprocess input text
        features = preprocess_text(text)

        # Debug information
        print("Feature type:", type(features))
        print("Feature shape:", features.shape)

        # Predict sentiment
        prediction = model.predict(features)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            "text": text,
            "sentiment": sentiment
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500


# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
