

from flask import Flask, request, jsonify
from flask_cors import CORS
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask_caching import Cache
import hashlib

import time

import os

app = Flask(__name__)
CORS(app)

app.config['CACHE_TYPE'] = 'SimpleCache' 
cache = Cache(app)

model_path = "tetianamohorian/hate_speech_model"
HISTORY_FILE = "history.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


def generate_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        text_hash = generate_text_hash(text)
        cached_result = cache.get(text_hash)
        if cached_result:
            return jsonify({"prediction": cached_result}), 200


        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)


        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()
        
        prediction_label = "Pravdepodobne toxický" if predictions == 1 else "Neutrálny text"
        
        cache.set(text_hash, prediction_label)
        
        entry = {
            "text": text,
            "prediction": prediction_label,
            "timestamp": datetime.now().isoformat()
        }
        
        response = app.response_class(
            response=json.dumps({"prediction": prediction_label}, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )
        
        entry = {
            "text": text,
            "prediction": prediction_label,
            "timestamp": datetime.now().isoformat()
        }

        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)



        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

