import os
import json
import torch
import torch.nn as nn
import time
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from torchcrf import CRF

# Import monitoring and explainability modules
from monitoring.metrics_logger import MetricsLogger
from monitoring.drift_detector import DriftDetector
from explainability.ner_explainer import NERExplainer
from explainability.retrieval_explainer import RetrievalExplainer

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Running on device: {'GPU' if DEVICE==0 else 'CPU'}")

# Load Vocabulary
print("Loading vocabulary...")
try:
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word2idx = vocab_data['word2idx']
    tag2idx = vocab_data['tag2idx']
    idx2tag = {int(k): v for k, v in vocab_data['idx2tag'].items()} # JSON keys are strings
    idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    print("[OK] Vocabulary loaded")
except FileNotFoundError:
    print("[ERROR] vocab.json not found! Run the notebook save cell first.")
    exit(1)

# Load Corpus
print("Loading corpus...")
try:
    with open('corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    print(f"[OK] Corpus loaded ({len(corpus)} entries)")
except FileNotFoundError:
    print("[ERROR] corpus.json not found! Run the notebook save cell first.")
    exit(1)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, embeddings=None):
        super().__init__()
        # We don't need to load pretrained embeddings here for inference if we load state_dict
        # But the architecture must match.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            mask = sentences != 0
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            mask = sentences != 0
            return self.crf.decode(emissions, mask=mask)

# Initialize Model
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
model = BiLSTM_NER(len(word2idx), len(tag2idx), EMBEDDING_DIM, HIDDEN_DIM)

# Load Weights
print("Loading NER model...")
try:
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("[OK] NER model loaded")
except FileNotFoundError:
    print("[ERROR] model.pth not found! Run the notebook save cell first.")
    exit(1)

# Load Sentence Transformer for Retrieval
print("Loading Sentence Transformer...")
retriever = SentenceTransformer('all-MiniLM-L6-v2')
# Pre-compute corpus embeddings
corpus_texts = [doc['text'] for doc in corpus]
corpus_embeddings = retriever.encode(corpus_texts, convert_to_tensor=True)
print("[OK] Retriever ready")

# Load Text Generation Model (Flan-T5)
print("Loading Flan-T5 (this may take a moment)...")
try:
    # Try loading a smaller model first to ensure stability
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small", 
        device=DEVICE
    )
    print("[OK] Generator ready (using flan-t5-small)")
except Exception as e:
    print(f"[WARNING] Failed to load generator: {e}")
    print("Switching to extractive-only mode.")
    generator = None

# Initialize Monitoring and Explainability
print("Initializing monitoring and explainability...")
metrics_logger = MetricsLogger()
drift_detector = DriftDetector(metrics_logger)
ner_explainer = NERExplainer(model, word2idx, idx2tag)
retrieval_explainer = RetrievalExplainer(retriever)
print("[OK] Monitoring and explainability ready")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def retrieve_docs(query, top_k=3):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))
    
    results = []
    scores = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append(corpus[idx.item()])
        scores.append(score.item())
    return results, scores

def detect_severity(user_input, entities, retrieved_docs):
    # Keywords
    urgent_keywords = ['bleeding', 'severe pain', 'infection', 'pus', 'swelling', 'fever', 'spreading']
    medical_keywords = ['cystic', 'rosacea', 'eczema', 'dermatitis', 'hormonal', 'persistent']
    
    # Check input
    if any(k in user_input.lower() for k in urgent_keywords):
        return "urgent", "⚠️ URGENT: Please see a doctor immediately. This requires medical attention.", False
        
    # Check retrieved docs for warnings
    for doc in retrieved_docs:
        if 'severity' in doc:
            if doc['severity'] == 'requires_medical':
                return "requires_medical", f"⚠️ Medical Evaluation Recommended: {doc.get('warning', 'See a dermatologist.')}", True
            if doc['severity'] == 'urgent':
                return "urgent", f"⚠️ URGENT: {doc.get('warning', 'Seek immediate care.')}", False
                
    return "mild", None, True

def generate_advice_extractive(user_query, entities, retrieved_docs):
    entity_text = ', '.join(entities) if entities else 'your concern'
    response = f"For {entity_text}:\\n\\n"
    
    actionable_keywords = ['use', 'apply', 'avoid', 'try', 'should', 'needs', 'requires', 'benefit', 'look for']
    recommendations = []
    
    for doc in retrieved_docs[:2]:
        text = doc['text']
        sentences = text.split('.')
        for sent in sentences:
            if any(keyword in sent.lower() for keyword in actionable_keywords):
                sent = sent.strip()
                if len(sent) > 20 and sent not in recommendations:
                    recommendations.append(sent)
                    if len(recommendations) >= 3:
                        break
                        
    if recommendations:
        for rec in recommendations:
            response += f"• {rec}.\\n"
    else:
        response += retrieved_docs[0]['text']
        
    response += f"\\n💡 Based on clinical evidence for {entity_text}."
    
    # Add sources
    response += "\\n\\n📚 SOURCES:\\n"
    for i, doc in enumerate(retrieved_docs, 1):
        response += f"[{i}] {doc.get('source', 'Medical Database')} ({doc.get('evidence_level', 'General')})\\n"
        
    return response

# ==========================================
# 4. ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('query', '')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    print(f"Processing: {user_input}")
    start_time = time.time()
    
    # 1. NER
    words = user_input.lower().split()
    word_indices = [word2idx.get(word, word2idx.get('<UNK>', 1)) for word in words]
    max_len = 25
    if len(word_indices) < max_len:
        word_indices = word_indices + [0] * (max_len - len(word_indices))
    else:
        word_indices = word_indices[:max_len]
        
    sentence_tensor = torch.LongTensor(word_indices).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(sentence_tensor)[0]
        
    # Extract Entities
    entities = []
    current_entity = []
    for word_idx, tag_idx in zip(word_indices, predictions):
        if word_idx == 0: break
        word = idx2word[word_idx]
        tag = idx2tag[tag_idx]
        
        if tag.startswith('B-'):
            if current_entity: entities.append(' '.join(current_entity))
            current_entity = [word]
        elif tag.startswith('I-') and current_entity:
            current_entity.append(word)
        elif tag == 'O':
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
        else:
            if current_entity: entities.append(' '.join(current_entity))
            current_entity = [word]
            
    if current_entity: entities.append(' '.join(current_entity))
    
    # 2. Retrieval
    query = ' '.join(entities) if entities else user_input
    docs, scores = retrieve_docs(query)
    
    # 3. Severity
    severity, warning, should_proceed = detect_severity(user_input, entities, docs)
    
    # 4. Generation
    if severity == 'urgent':
        advice = warning
    else:
        # Use Extractive for reliability (or switch to generator if preferred)
        # advice = generate_advice_extractive(user_input, entities, docs)
        
        # Let's use the Generator for "wow" factor if safe
        context_text = "\n".join([f"Source {i+1}: {d['text']}" for i, d in enumerate(docs)])
        entity_text = ', '.join(entities) if entities else 'skincare issue'
        prompt = f"""Answer this skincare question using the medical sources below.

Question: What should I do about {entity_text}?

Medical Sources:
{context_text}

Answer: Based on these sources, for {entity_text}, you should"""

        try:
            gen_text = generator(
                prompt, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.5
            )[0]['generated_text']
            
            # Cleanup
            if "Answer:" in gen_text: gen_text = gen_text.split("Answer:")[-1].strip()
            advice = gen_text
            
            # Append sources
            advice += "\n\n📚 SOURCES:\n"
            for i, doc in enumerate(docs, 1):
                advice += f"[{i}] {doc.get('source', 'Medical Database')}\n"
                
            if warning:
                advice = f"{warning}\n\n{advice}"
                
        except Exception as e:
            print(f"Generation failed: {e}")
            advice = generate_advice_extractive(user_input, entities, docs)

    # Log metrics
    response_time = time.time() - start_time
    confidence_level = 'high' if max(scores) > 0.6 else 'low'
    
    try:
        metrics_logger.log_prediction(
            query=user_input,
            entities=entities,
            severity=severity,
            confidence=max(scores),
            response_time=response_time,
            top_score=max(scores)
        )
    except Exception as e:
        print(f"Failed to log metrics: {e}")
    
    return jsonify({
        'advice': advice,
        'entities': entities,
        'severity': severity,
        'confidence': confidence_level,
        'query_id': user_input[:20]  # For explanation lookup
    })

@app.route('/explain', methods=['POST'])
def explain():
    """Generate explanation for a prediction."""
    data = request.json
    user_input = data.get('query', '')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Re-run NER for explanation
    words = user_input.lower().split()
    word_indices = [word2idx.get(word, word2idx.get('<UNK>', 1)) for word in words]
    max_len = 25
    if len(word_indices) < max_len:
        word_indices = word_indices + [0] * (max_len - len(word_indices))
    else:
        word_indices = word_indices[:max_len]
    
    sentence_tensor = torch.LongTensor(word_indices).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(sentence_tensor)[0]
    
    # Extract entities
    entities = []
    current_entity = []
    for word_idx, tag_idx in zip(word_indices, predictions):
        if word_idx == 0: break
        word = idx2word[word_idx]
        tag = idx2tag[tag_idx]
        
        if tag.startswith('B-'):
            if current_entity: entities.append(' '.join(current_entity))
            current_entity = [word]
        elif tag.startswith('I-') and current_entity:
            current_entity.append(word)
        elif tag == 'O':
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
        else:
            if current_entity: entities.append(' '.join(current_entity))
            current_entity = [word]
    
    if current_entity: entities.append(' '.join(current_entity))
    
    try:
        # Get NER explanation
        ner_explanation = ner_explainer.explain_prediction(sentence_tensor, words[:len(predictions)])
        
        # Get retrieval explanation
        query = ' '.join(entities) if entities else user_input
        docs, scores = retrieve_docs(query)
        retrieval_explanation = retrieval_explainer.explain_retrieval(query, docs, scores)
        confidence_breakdown = retrieval_explainer.get_confidence_breakdown(scores)
        
        return jsonify({
            'ner_explanation': ner_explanation,
            'retrieval_explanation': retrieval_explanation,
            'confidence_breakdown': confidence_breakdown
        })
    except Exception as e:
        import traceback
        with open('error_log.txt', 'w') as f:
            traceback.print_exc(file=f)
        return jsonify({'error': str(e)}), 500

@app.route('/drift_status', methods=['GET'])
def drift_status():
    """Get current drift detection status."""
    try:
        if not drift_detector.baseline_data:
            drift_detector.set_baseline()
        
        drift_results = drift_detector.detect_drift()
        summary = drift_detector.get_drift_summary()
        
        return jsonify({
            'summary': summary,
            'details': drift_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" DermaAI Backend Starting...")
    print("="*50)
    print(" Main App: http://localhost:5001")
    print(" Dashboard: Run 'python dashboard.py' for monitoring")
    print("="*50 + "\n")
    app.run(debug=False, port=5001)
