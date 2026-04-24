from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# path to my saved trained model
MODEL_PATH = "mentalbert_classifier"

# load tokenizer and trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# numerical label mapping
id2label = {0: "anxiety",
    1: "depression",
    2: "neutral"}

# function to process the input text for predictions
def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256)

# disable gradient calculation (so no learning occurs)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    return {
        "prediction": id2label[pred_class.item()],
        "confidence": float(confidence.item()),
        "probabilities": probs.squeeze().tolist()
    }

# homepage route where flask sends my html file to the browser
@app.route("/")
def home():
    return render_template("index.html")

# endpoint that recieves input text from frontend, processes and returns predicted result
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")
#handle empty inputs
    if not text:
        return jsonify({"error": "No text provided"})
# get model pred and return result a s a json response
    result = predict_text(text)
    return jsonify(result)

# run flask app finally
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)