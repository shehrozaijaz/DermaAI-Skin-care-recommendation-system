# 🌟 DermaAI - Intelligent Skincare Recommendation System

An AI-powered skincare recommendation system that combines **Named Entity Recognition (NER)**, **Retrieval-Augmented Generation (RAG)**, and **Explainable AI (XAI)** to provide personalized, evidence-based skincare advice with real-time monitoring and drift detection.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [File Descriptions](#file-descriptions)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Data Drift Monitoring](#data-drift-monitoring)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**DermaAI** is an intelligent skincare recommendation system designed to provide personalized, evidence-based skincare advice. The system analyzes user queries about skin concerns (acne, hyperpigmentation, aging, etc.) and retrieves relevant treatment recommendations from a curated medical knowledge base.

### Purpose

- **Democratize Skincare Knowledge**: Provide accessible, evidence-based skincare advice
- **Personalized Recommendations**: Tailor advice based on specific skin concerns and types
- **Transparency**: Explain why certain recommendations are made (XAI)
- **Quality Assurance**: Monitor model performance and detect data drift over time

---

## ✨ Key Features

### 1. **Named Entity Recognition (NER)**
- Custom BiLSTM-CRF model trained to extract skincare entities
- Identifies: `CONDITION`, `SYMPTOM`, `TREATMENT`, `PRODUCT`, `INGREDIENT`
- Trained on domain-specific skincare data

### 2. **Retrieval-Augmented Generation (RAG)**
- Semantic search using Sentence-BERT embeddings
- Retrieves top-K relevant documents from medical corpus
- Combines retrieval with Flan-T5 text generation
- Provides source citations for transparency

### 3. **Explainable AI (XAI)**
- **Token Importance**: Gradient-based attribution showing which words influenced NER predictions
- **Retrieval Explanation**: Shows why specific documents were retrieved (semantic similarity scores)
- **Confidence Breakdown**: Displays overall confidence with component-level scores

### 4. **Data Drift Monitoring**
- Tracks prediction distributions over time
- Kolmogorov-Smirnov test for statistical drift detection
- Real-time dashboard with Plotly visualizations
- SQLite database for metrics logging

### 5. **Interactive Web Interface**
- Clean, modern UI with glassmorphism design
- Real-time chat interface
- Explanation modal for transparency
- Responsive design for all devices

---

## 🏗️ Project Architecture

```
┌─────────────────┐
│   User Query    │
│ "I have acne"   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Flask Backend (app.py)          │
│  ┌───────────────────────────────────┐  │
│  │  1. NER Model (BiLSTM-CRF)        │  │
│  │     Extracts: ["acne"]            │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │  2. Retriever (Sentence-BERT)     │  │
│  │     Finds relevant corpus docs    │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │  3. Generator (Flan-T5)           │  │
│  │     Synthesizes response          │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │  4. Metrics Logger                │  │
│  │     Saves to SQLite               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Response with Citations             │
│  "For acne, use salicylic acid 2%..."   │
│  Sources: [1] AAD 2024, [2] JAMA 2023   │
└─────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### **Backend**
- **Flask 3.0.0**: Web framework
- **PyTorch**: Deep learning framework
- **TorchCRF**: Conditional Random Fields for NER
- **Transformers (Hugging Face)**: Flan-T5 text generation
- **Sentence-Transformers**: Semantic embeddings

### **Frontend**
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Fetch API**: Asynchronous requests
- **Glassmorphism Design**: Modern UI aesthetics

### **Monitoring & Visualization**
- **Dash**: Interactive dashboard framework
- **Plotly**: Data visualization
- **SQLite**: Metrics storage
- **SciPy**: Statistical tests (KS test)

### **NLP & Embeddings**
- **GloVe**: Pre-trained word embeddings (100d)
- **Sentence-BERT**: Semantic sentence embeddings
- **BiLSTM-CRF**: Sequence labeling architecture

---

## 📁 Project Structure

```
skin-care-recommendation-system-main/
│
├── app.py                          # Main Flask application
├── dashboard.py                    # Monitoring dashboard (Dash)
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── model.pth                   # Trained BiLSTM-CRF NER model
│   ├── vocab.json                  # Vocabulary mapping
│   └── corpus.json                 # Medical knowledge base (150 entries)
│
├── data/
│   ├── dl_data.csv                 # NER training data
│   └── glove.6B.100d.txt          # GloVe embeddings (347MB)
│
├── monitoring/
│   ├── __init__.py
│   ├── metrics_logger.py           # SQLite logging
│   └── drift_detector.py           # KS test drift detection
│
├── explainability/
│   ├── __init__.py
│   ├── ner_explainer.py            # Token importance (gradients)
│   └── retrieval_explainer.py      # Document relevance scores
│
├── templates/
│   └── index.html                  # Frontend HTML
│
├── static/
│   ├── style.css                   # Glassmorphism styling
│   └── script.js                   # Frontend logic
│
├── notebooks/
│   └── main_flan_t5_large_cleaned.ipynb  # Model training notebook
│
└── README.md                       # This file
```

---

## ⚙️ How It Works

### **End-to-End Flow**

1. **User Input** → User types query: *"I have dark skin and acne"*

2. **Named Entity Recognition (NER)**
   - BiLSTM-CRF model processes query
   - Extracts entities: `["dark skin", "acne"]`
   - Tags: `CONDITION`, `SYMPTOM`

3. **Semantic Retrieval**
   - Sentence-BERT encodes query + entities
   - Computes cosine similarity with corpus embeddings
   - Retrieves top-3 most relevant documents

4. **Response Generation**
   - Flan-T5 model synthesizes response from retrieved docs
   - Combines multiple sources into coherent advice
   - Adds citations with source attribution

5. **Metrics Logging**
   - Saves query, entities, confidence to SQLite
   - Tracks timestamp, response time, model version

6. **Drift Detection**
   - Compares current distribution vs. baseline
   - KS test p-value < 0.05 triggers drift alert
   - Dashboard visualizes trends over time

---

## 🚀 Setup Instructions

### **Prerequisites**
- Python 3.13
- 8GB+ RAM (for model loading)
- 2GB+ disk space

### **Step 1: Clone Repository**
```bash
git clone <repository-url>
cd skin-care-recommendation-system-main
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `flask==3.0.0`
- `torch`
- `torchcrf`
- `transformers`
- `sentence-transformers`
- `dash`
- `plotly`
- `scipy`
- `scikit-learn`
- `numpy`
- `pandas`
- `tf-keras`
- `tensorflow`

### **Step 3: Download GloVe Embeddings**
Download `glove.6B.100d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and place in project root.

### **Step 4: Train Model (Optional)**
If you want to retrain the model:
```bash
# Open Jupyter notebook
jupyter notebook main_flan_t5_large_cleaned.ipynb

# Run all cells to:
# - Load and preprocess data
# - Train BiLSTM-CRF NER model
# - Create corpus embeddings
# - Save model.pth, vocab.json, corpus.json
```

### **Step 5: Run Application**
```bash
# Start Flask app
python app.py

# Access at: http://localhost:5001
```

### **Step 6: Run Monitoring Dashboard (Optional)**
```bash
# In a separate terminal
python dashboard.py

# Access at: http://localhost:8050
```

---

## 📖 Usage Guide

### **Basic Usage**

1. **Open Browser**: Navigate to `http://localhost:5001`

2. **Enter Query**: Type your skin concern
   - Example: *"I have oily skin and large pores"*

3. **View Response**: Get personalized recommendations with citations

4. **Explain**: Click "Explain" button to see:
   - Which words were important for NER
   - Why specific documents were retrieved
   - Confidence breakdown

### **Advanced Features**

#### **Monitoring Dashboard**
- View prediction trends over time
- Check drift detection status
- Analyze confidence score distributions
- Export metrics for analysis

#### **API Usage**
```python
import requests

# Get recommendation
response = requests.post('http://localhost:5001/predict', 
    json={'user_input': 'I have acne'})
print(response.json())

# Get explanation
explanation = requests.post('http://localhost:5001/explain',
    json={'query': 'I have acne'})
print(explanation.json())
```

---

## 📄 File Descriptions

### **Core Application Files**

#### **`app.py`** (Main Backend)
**Purpose**: Flask web server handling all API requests and model inference

**Key Components**:
- `BiLSTM_NER` class: Custom NER model architecture
- Model loading: Loads `model.pth`, `vocab.json`, `corpus.json`
- Sentence-BERT retriever: Semantic search
- Flan-T5 generator: Text generation (fallback to extractive mode)
- Metrics logger integration
- Explainability module integration

**Routes**:
- `/` - Serve frontend
- `/predict` - Main recommendation endpoint
- `/explain` - XAI explanations
- `/drift_status` - Drift detection status

**Workflow**:
```python
1. Load models (NER, Sentence-BERT, Flan-T5)
2. Receive user query via /predict
3. Run NER to extract entities
4. Retrieve top-K documents using embeddings
5. Generate response (extractive summary)
6. Log metrics to SQLite
7. Return response with citations
```

#### **`dashboard.py`** (Monitoring Dashboard)
**Purpose**: Real-time visualization of model performance and drift detection

**Features**:
- **Prediction Timeline**: Line chart of predictions over time
- **Confidence Distribution**: Histogram of confidence scores
- **Entity Frequency**: Bar chart of most common entities
- **Drift Status**: Alert banner if drift detected
- **Auto-refresh**: Updates every 30 seconds

**Technology**: Plotly Dash with Bootstrap components

**Drift Detection**:
```python
1. Load baseline distribution (first 100 predictions)
2. Compare with recent predictions (last 100)
3. Run Kolmogorov-Smirnov test
4. If p-value < 0.05: DRIFT DETECTED
5. Display alert on dashboard
```

### **Monitoring Module**

#### **`monitoring/metrics_logger.py`**
**Purpose**: Log all predictions to SQLite database

**Schema**:
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    query TEXT,
    entities TEXT (JSON),
    confidence REAL,
    response_time REAL,
    model_version TEXT
)
```

**Methods**:
- `log_prediction()`: Save prediction to DB
- `get_recent_predictions()`: Retrieve last N predictions
- `get_prediction_stats()`: Aggregate statistics

#### **`monitoring/drift_detector.py`**
**Purpose**: Detect distribution shifts in model predictions

**Algorithm**:
```python
1. Extract confidence scores from baseline (first 100)
2. Extract confidence scores from current (last 100)
3. Run scipy.stats.ks_2samp(baseline, current)
4. Return: {
    'drift_detected': p_value < 0.05,
    'p_value': p_value,
    'statistic': ks_statistic
}
```

**Use Case**: Alerts when model behavior changes (e.g., new data patterns, model degradation)

### **Explainability Module**

#### **`explainability/ner_explainer.py`**
**Purpose**: Explain NER predictions using gradient-based attribution

**Method**: Integrated Gradients
```python
1. Get model embeddings for input tokens
2. Compute gradients w.r.t. embeddings
3. Calculate importance = gradient × embedding
4. Normalize to [0, 1] range
5. Return token-importance pairs
```

**Output**:
```json
{
    "tokens": ["I", "have", "dark", "skin"],
    "importance": [0.1, 0.2, 0.9, 0.8],
    "entities": [
        {"word": "dark skin", "tag": "CONDITION", "importance": 0.85}
    ]
}
```

#### **`explainability/retrieval_explainer.py`**
**Purpose**: Explain why documents were retrieved

**Method**: Semantic similarity decomposition
```python
1. Compute query embedding
2. Compute document embeddings
3. Calculate cosine similarity scores
4. Extract matching keywords
5. Return ranked explanations
```

**Output**:
```json
{
    "documents": [
        {
            "text": "Dark skin hyperpigmentation...",
            "score": 0.87,
            "keywords": ["dark skin", "hyperpigmentation"],
            "reason": "High semantic match (87%)"
        }
    ]
}
```

### **Frontend Files**

#### **`templates/index.html`**
**Purpose**: Main user interface

**Features**:
- Chat interface with message bubbles
- Input field with send button
- Explanation modal (popup)
- Responsive design

#### **`static/style.css`**
**Purpose**: Modern glassmorphism styling

**Design Elements**:
- Soft pastel gradient background
- Frosted glass effect (backdrop-filter)
- Smooth animations and transitions
- Token highlighting in explanations

#### **`static/script.js`**
**Purpose**: Frontend logic and API communication

**Functions**:
- `sendMessage()`: Send query to `/predict`
- `showExplanation()`: Fetch and display XAI data
- `appendMessage()`: Render chat messages
- `formatCitations()`: Parse and display sources

---

## 🔍 Explainable AI (XAI)

### **Why XAI Matters**
- **Trust**: Users need to understand why recommendations are made
- **Safety**: Medical advice requires transparency
- **Debugging**: Helps identify model errors

### **XAI Components**

#### **1. Token Importance (NER)**
Shows which words influenced entity extraction:
```
Query: "I have dark skin and acne"
Importance:
  I        ▓░░░░░░░░░ 10%
  have     ▓░░░░░░░░░ 15%
  dark     ▓▓▓▓▓▓▓▓▓░ 90%  ← Important!
  skin     ▓▓▓▓▓▓▓▓░░ 85%  ← Important!
  and      ▓░░░░░░░░░ 12%
  acne     ▓▓▓▓▓▓▓▓▓▓ 95%  ← Important!
```

#### **2. Retrieval Explanation**
Shows why documents were selected:
```
Document 1 (Score: 87%)
  "Dark skin hyperpigmentation requires..."
  Keywords: dark skin, hyperpigmentation
  Reason: High semantic match based on "dark skin"

Document 2 (Score: 82%)
  "For acne, use salicylic acid 2%..."
  Keywords: acne, treatment
  Reason: Matches "acne" with treatment focus
```

#### **3. Confidence Breakdown**
```
Overall Confidence: 78%
├─ NER Confidence: 85%
├─ Retrieval Confidence: 75%
└─ Generation Confidence: 74%
```

---

## 📊 Data Drift Monitoring

### **What is Data Drift?**
Data drift occurs when the distribution of input data changes over time, potentially degrading model performance.

### **Detection Method**

**Kolmogorov-Smirnov (KS) Test**:
- Compares two distributions (baseline vs. current)
- Null hypothesis: Distributions are the same
- If p-value < 0.05: Reject null → **Drift detected**

### **Monitoring Workflow**

```python
# 1. Baseline (first 100 predictions)
baseline_confidences = [0.75, 0.82, 0.79, ...]

# 2. Current (last 100 predictions)
current_confidences = [0.65, 0.58, 0.62, ...]  # Lower!

# 3. KS Test
statistic, p_value = ks_2samp(baseline, current)

# 4. Alert if drift
if p_value < 0.05:
    print("⚠️ DRIFT DETECTED - Model retraining recommended")
```

### **Dashboard Metrics**
- **Prediction Count**: Total predictions logged
- **Average Confidence**: Mean confidence over time
- **Drift Status**: Green (OK) / Red (Drift detected)
- **Entity Distribution**: Most common extracted entities

### **When to Retrain**
- Drift detected for > 1 week
- Average confidence drops > 10%
- User feedback indicates poor quality

---

## 🧪 Model Training

### **Training Pipeline** (Jupyter Notebook)

#### **1. Data Preparation**
```python
# Load training data
df = pd.read_csv('dl_data.csv')

# Format: sentence, word, tag
# Example: "I have acne", "acne", "CONDITION"
```

#### **2. GloVe Embeddings**
```python
# Load GloVe (memory-efficient)
embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        word, *vector = line.split()
        if word in vocabulary:  # Only load needed words
            embeddings_index[word] = np.array(vector, dtype='float32')
```

#### **3. Model Architecture**
```python
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
```

#### **4. Training**
```python
# Train for 50 epochs
for epoch in range(50):
    for batch in train_loader:
        optimizer.zero_grad()
        emissions = model(batch['input'])
        loss = -model.crf(emissions, batch['tags'], mask=batch['mask'])
        loss.backward()
        optimizer.step()
```

#### **5. Save Artifacts**
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Save vocabulary
with open('vocab.json', 'w') as f:
    json.dump(word2idx, f)

# Save corpus
with open('corpus.json', 'w') as f:
    json.dump(enhanced_corpus, f)
```

### **Training Data Format**

**`dl_data.csv`**:
```csv
sentence,word,tag
"I have acne",I,O
"I have acne",have,O
"I have acne",acne,B-CONDITION
"Use retinol for wrinkles",Use,O
"Use retinol for wrinkles",retinol,B-TREATMENT
"Use retinol for wrinkles",for,O
"Use retinol for wrinkles",wrinkles,B-SYMPTOM
```

**Tags**:
- `B-CONDITION`: Beginning of condition (e.g., "acne")
- `I-CONDITION`: Inside condition (e.g., "dark skin")
- `B-SYMPTOM`: Beginning of symptom
- `B-TREATMENT`: Beginning of treatment
- `O`: Outside any entity

---

## 🌐 API Endpoints

### **1. GET `/`**
**Description**: Serve frontend HTML

**Response**: `index.html`

---

### **2. POST `/predict`**
**Description**: Get skincare recommendation

**Request**:
```json
{
    "user_input": "I have oily skin and large pores"
}
```

**Response**:
```json
{
    "response": "For oily skin with large pores, use niacinamide 5% serum to regulate sebum, salicylic acid 2% cleanser, and clay mask 1-2 times weekly...",
    "entities": [
        {"word": "oily skin", "tag": "CONDITION"},
        {"word": "large pores", "tag": "SYMPTOM"}
    ],
    "sources": [
        "[1] American Academy of Dermatology, 2024",
        "[2] Journal of Cosmetic Dermatology, 2023"
    ],
    "confidence": 0.82
}
```

---

### **3. POST `/explain`**
**Description**: Get XAI explanations

**Request**:
```json
{
    "query": "I have dark skin and acne"
}
```

**Response**:
```json
{
    "ner_explanation": {
        "tokens": ["I", "have", "dark", "skin", "and", "acne"],
        "importance": [0.1, 0.15, 0.9, 0.85, 0.12, 0.95],
        "entities": [
            {"word": "dark skin", "tag": "CONDITION", "importance": 0.875},
            {"word": "acne", "tag": "CONDITION", "importance": 0.95}
        ]
    },
    "retrieval_explanation": {
        "documents": [
            {
                "text": "Dark skin hyperpigmentation requires...",
                "score": 0.87,
                "keywords": ["dark skin", "hyperpigmentation"]
            }
        ]
    },
    "confidence_breakdown": {
        "overall": 0.78,
        "ner": 0.85,
        "retrieval": 0.75,
        "generation": 0.74
    }
}
```

---

### **4. GET `/drift_status`**
**Description**: Check data drift status

**Response**:
```json
{
    "drift_detected": false,
    "p_value": 0.23,
    "statistic": 0.12,
    "baseline_size": 100,
    "current_size": 100,
    "message": "No drift detected"
}
```

---

## 🚀 Future Improvements

### **Short-term**
- [ ] Add user feedback (thumbs up/down)
- [ ] Expand corpus to 500+ entries
- [ ] Fine-tune BERT for medical NER
- [ ] Add multi-language support

### **Medium-term**
- [ ] Implement re-ranking with cross-encoder
- [ ] Add image upload for skin analysis
- [ ] Create mobile app (React Native)
- [ ] A/B testing framework

### **Long-term**
- [ ] Federated learning for privacy
- [ ] Integration with telemedicine platforms
- [ ] Personalized product recommendations
- [ ] Clinical trial integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Contribution Areas**
- **Corpus Expansion**: Add more medical entries
- **Model Improvements**: Better NER/generation models
- **UI/UX**: Enhance frontend design
- **Documentation**: Improve guides and tutorials

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **GloVe Embeddings**: [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- **Sentence-BERT**: [UKPLab](https://www.sbert.net/)
- **Flan-T5**: [Google Research](https://huggingface.co/google/flan-t5-small)
- **Medical Sources**: American Academy of Dermatology, JAMA Dermatology, British Journal of Dermatology

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for better skincare through AI**
