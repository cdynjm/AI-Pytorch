import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')

# Tokenizer & stemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Neural network class (same as training)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Load trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load intents for responses
with open('intents.json', 'r') as f:
    intents = json.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Chatbot is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    input_text = data['text']

    tokenized = tokenize(input_text)
    bow = bag_of_words(tokenized, all_words)
    bow = torch.from_numpy(bow).float().unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(bow)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

    if prob.item() > 0.50:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = np.random.choice(intent['responses'])
                return jsonify({"response": response, "tag": tag, "probability": prob.item()})
    else:
        return jsonify({"response": "Sorry, I do not understand.", "tag": "unknown", "probability": prob.item()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
