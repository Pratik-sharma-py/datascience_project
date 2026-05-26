from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

try:
    # Load trained models
    print("Loading models...")
    cv = pickle.load(open(r'notebooks\cv.pkl', 'rb'))
    clf = pickle.load(open(r'notebooks\clf.pkl', 'rb'))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

ps = PorterStemmer()

def clean_and_stem(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    stemmed = [ps.stem(word) for word in words]
    return " ".join(stemmed)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({'error': 'Please enter email text'}), 400
        
        cleaned = clean_and_stem(email_text)
        vector = cv.transform([cleaned])
        
        prediction = clf.predict(vector)[0]
        probability = clf.predict_proba(vector)[0]
        
        if prediction == 1:
            result = "HAM"
            confidence = float(probability[1])
        else:
            result = "SPAM"
            confidence = float(probability[0])
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        print("Starting Flask app on http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask: {e}")