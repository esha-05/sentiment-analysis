# SentimentAI — Real-Time Sentiment Analysis

A full-stack sentiment analysis application with an LSTM-based deep learning model,
Flask REST API backend, and interactive Streamlit/HTML frontend.
---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install flask scikit-learn numpy matplotlib seaborn
# For Streamlit frontend:
pip install streamlit plotly pandas
# For TensorFlow (optional, production):
pip install tensorflow
```

### Train the model
```bash
cd sentiment_app
python3 -m model.train_model
# → Trains on 6000 synthetic + noisy samples
# → Saves artifacts to model/saved/
# → Generates evaluation_charts.png
```

### Start the Flask API
```bash
cd sentiment_app
python3 backend/app.py
# → Starts on http://localhost:5000
```

### Open the Frontend
```bash
# Option A: HTML (no install needed)
open frontend/index.html

# Option B: Streamlit
streamlit run frontend/streamlit_app.py
```

---

## 🌐 API Endpoints

| Method | Path                   | Description                        |
|--------|------------------------|------------------------------------|
| POST   | `/api/predict`         | Single text prediction             |
| POST   | `/api/predict/batch`   | Batch prediction (≤50 texts)       |
| GET    | `/api/health`          | Health check + model warm-up       |
| GET    | `/api/metrics`         | Model performance metrics          |
| GET    | `/api/model/info`      | Architecture & config metadata     |

### POST /api/predict

**Request:**
```json
{
  "text": "This product is absolutely amazing!"
}
```

**Response:**
```json
{
  "success": true,
  "text": "This product is absolutely amazing!",
  "prediction": {
    "label": "Positive",
    "label_index": 2,
    "confidence": 0.9267,
    "probabilities": {
      "Negative": 0.0312,
      "Neutral":  0.0421,
      "Positive": 0.9267
    },
    "text_length": 35
  },
  "latency_ms": 12.4
}
```

**Error Responses:**
```json
{ "success": false, "error": "'text' must not be empty.", "code": 400 }
{ "success": false, "error": "Content-Type must be application/json", "code": 415 }
{ "success": false, "error": "'text' exceeds 5000 character limit.", "code": 413 }
```

### POST /api/predict/batch

**Request:**
```json
{
  "texts": [
    "Great product, love it!",
    "Terrible, broken on arrival.",
    "Just okay, nothing special."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [
    { "index": 0, "text": "Great product, love it!", "prediction": {...} },
    { "index": 1, "text": "Terrible, broken on arrival.", "prediction": {...} },
    { "index": 2, "text": "Just okay, nothing special.", "prediction": {...} }
  ],
  "error_count": 0,
  "latency_ms": 28.7
}
```

**Training Config:**
- Optimizer: Adam (lr=0.001, adaptive)
- Batch size: 64
- Early stopping: 10 epochs patience
- Validation split: 10%
- Max iterations: 200

---

## 📊 Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 90.00%  |
| Precision | 90.23%  |
| Recall    | 90.00%  |
| F1-Score  | 89.98%  |

**Per-Class Metrics:**

| Class    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| Negative | 91.97%    | 85.94% | 88.85%| 192     |
| Neutral  | 91.89%    | 87.95% | 89.88%| 224     |
| Positive | 86.17%    | 95.65% | 90.67%| 184     |
--

## 🧪 API Testing (Postman)

Import the following collection or test manually:

```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product, it is amazing!"}'

# Batch prediction
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "Okay."]}'

# Model metrics
curl http://localhost:5000/api/metrics

# Error test (empty text)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

Run automated test suite:
## 🚀 Deployment

### Local
```bash
python3 backend/app.py
# → http://localhost:5000
```

## 📋 Configuration

| Parameter  | Value  | Location            |
|------------|--------|---------------------|
| VOCAB_SIZE | 5000   | sentiment_core.py   |
| MAX_LEN    | 50     | sentiment_core.py   |
| EMBED_DIM  | 64     | sentiment_core.py   |
| API_PORT   | 5000   | backend/app.py      |
| LOG_FILE   | logs/api.log | backend/app.py  |
| MAX_TEXT   | 5000 chars | backend/app.py  |
| MAX_BATCH  | 50 texts   | backend/app.py  |

---

## 📝 Logs

All API requests are logged to `logs/api.log`:
```
2025-03-30 12:01:23,456 [INFO] POST /api/predict → 200 (12.4 ms)
2025-03-30 12:01:30,789 [INFO] POST /api/predict → 400 (0.1 ms)
2025-03-30 12:01:45,123 [ERROR] Prediction error: ...
```

---

## 🏗 Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Model     | sklearn MLPClassifier + Custom Embedding|
| Framework | Python 3.11, NumPy, scikit-learn        |
| Backend   | Flask 3.x, REST API, JSON              |
| Frontend  | HTML5 + Vanilla JS (standalone)         |
| Alt UI    | Streamlit + Plotly                      |
| Testing   | Flask test client, 54 unit tests        |
| Logging   | Python logging (file + stdout)          |
