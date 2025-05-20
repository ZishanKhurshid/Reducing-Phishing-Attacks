from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import tldextract
from urllib.parse import urlparse

# Load model and feature extractor
model = joblib.load('phishing_model.pkl')

# Define feature extraction logic
def extract_features(df):
    df = df.copy()
    df['URL'] = df['URL'].astype(str)
    df['url_length'] = df['URL'].str.len()
    df['num_special_chars'] = df['URL'].str.count(r'[^a-zA-Z0-9]')
    
    def get_domain_parts(url):
        try:
            ext = tldextract.extract(str(url))
            return pd.Series({
                'subdomain': ext.subdomain,
                'domain': ext.domain,
                'suffix': ext.suffix,
                'is_ip': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', ext.domain) else 0
            })
        except:
            return pd.Series({
                'subdomain': None,
                'domain': None,
                'suffix': None,
                'is_ip': 0
            })

    domain_parts = df['URL'].apply(get_domain_parts)
    df = pd.concat([df, domain_parts], axis=1)
    
    df['subdomain_length'] = df['subdomain'].astype(str).str.len().fillna(0)
    df['has_https'] = df['URL'].str.startswith('https').astype(int)
    
    return df

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Phishing URL Detection API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "Missing URL"}), 400

        df = pd.DataFrame({"URL": [url]})
        features = extract_features(df)
        features = features.drop(['URL', 'is_valid_url'], axis=1, errors='ignore')
        if 'label' in features.columns:
            features = features.drop('label', axis=1)

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        return jsonify({
            "url": url,
            "prediction": "Phishing" if pred else "Legitimate",
            "confidence": round(float(proba[1] if pred else proba[0]), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
