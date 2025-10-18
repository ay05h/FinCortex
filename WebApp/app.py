from flask import Flask, render_template, request, jsonify
import torch
from models.financial_causal_detector import FinancialCausalDetector
from models.vocabulary import Vocabulary
import os

app = Flask(__name__)

# Load model and vocabulary
def load_model_and_vocab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary
    vocab = Vocabulary.load_vocab("models/financial_vocab.pkl")
    
    # Load model configuration and state
    checkpoint = torch.load("models/financial_causal_model.pt", map_location=device)
    
    # Initialize model with the same configuration
    model = FinancialCausalDetector(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, vocab, device

model, vocab, device = load_model_and_vocab()

def predict_causal_relation(text):
    """Predict cause and effect in a given text."""
    model.eval()
    
    # Tokenize and numericalize the text
    numeric_text = vocab.numericalize(text)
    text_tensor = torch.tensor(numeric_text).unsqueeze(0).to(device)  # [1, seq_len]
    text_length = torch.tensor([len(numeric_text)]).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(text_tensor, text_length)
    
    # Get predictions
    cause_start_logits = outputs["cause_start_logits"][0]
    effect_start_logits = outputs["effect_start_logits"][0]
    relation_logits = outputs["relation_logits"][0]
    
    # Get probabilities
    cause_start_probs = torch.sigmoid(cause_start_logits).cpu().numpy()
    effect_start_probs = torch.sigmoid(effect_start_logits).cpu().numpy()
    
    # Extract spans using a threshold
    threshold = 0.5
    cause_spans = []
    effect_spans = []
    
    # Find cause spans
    current_span = False
    span_start = 0
    for i, prob in enumerate(cause_start_probs):
        if prob > threshold and not current_span:
            current_span = True
            span_start = i
        elif prob <= threshold and current_span:
            current_span = False
            cause_spans.append((span_start, i-1))
    if current_span:
        cause_spans.append((span_start, len(cause_start_probs)-1))
    
    # Find effect spans
    current_span = False
    span_start = 0
    for i, prob in enumerate(effect_start_probs):
        if prob > threshold and not current_span:
            current_span = True
            span_start = i
        elif prob <= threshold and current_span:
            current_span = False
            effect_spans.append((span_start, i-1))
    if current_span:
        effect_spans.append((span_start, len(effect_start_probs)-1))
    
    # Get relation type
    relation_type = torch.argmax(relation_logits).item()
    relation_types = ["No relation", "Cause->Effect", "Effect->Cause"]
    relation_confidence = float(torch.softmax(relation_logits, dim=0).max())
    
    # Convert token indices back to words
    text_tokens = vocab.tokenize(text)
    
    causes = []
    for start, end in cause_spans:
        if start < len(text_tokens) and end < len(text_tokens):
            cause_text = " ".join(text_tokens[start:end+1])
            causes.append({
                "text": cause_text,
                "start": start,
                "end": end,
                "confidence": float(max(cause_start_probs[start:end+1])) if end < len(cause_start_probs) else 0.0
            })
    
    effects = []
    for start, end in effect_spans:
        if start < len(text_tokens) and end < len(text_tokens):
            effect_text = " ".join(text_tokens[start:end+1])
            effects.append({
                "text": effect_text,
                "start": start,
                "end": end,
                "confidence": float(max(effect_start_probs[start:end+1])) if end < len(effect_start_probs) else 0.0
            })
    
    # Calculate overall confidence
    cause_conf = max([c['confidence'] for c in causes]) if causes else 0.0
    effect_conf = max([e['confidence'] for e in effects]) if effects else 0.0
    
    return {
        "text": text,
        "tokens": text_tokens,
        "causes": causes,
        "effects": effects,
        "relation": {
            "type": relation_types[relation_type],
            "confidence": relation_confidence
        },
        "cause_confidence": cause_conf,
        "effect_confidence": effect_conf
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    
    if not text.strip():
        return jsonify({"error": "Please enter some text to analyze"}), 400
    
    try:
        result = predict_causal_relation(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)