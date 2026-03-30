# 🏠 HomeScout – AI-Powered Rental Listing Fraud Detection & Neighborhood Intelligence

> **An end-to-end Generative AI platform combining PyTorch deep learning, DistilBERT embeddings, FAISS vector search, and Groq LLM (Llama 3.3-70B) to detect fraudulent rental listings and generate explainable analyst reports in real time.**

---

## 📸 Screenshots

<img width="1909" height="970" alt="HS-1" src="https://github.com/user-attachments/assets/eb650031-a6d0-4e61-b215-f176bc171ad6" />
<img width="1914" height="972" alt="HS-2" src="https://github.com/user-attachments/assets/a97b3602-9471-465e-b578-52a6d74ed216" />
<img width="1900" height="957" alt="HS-3" src="https://github.com/user-attachments/assets/3f518a0b-0821-4079-a3b9-3fd21adbcbc4" />
<img width="1908" height="967" alt="HS-4" src="https://github.com/user-attachments/assets/c4351283-dbc8-4b77-ae39-579bc4f1de26" />
<img width="1906" height="968" alt="HS-5" src="https://github.com/user-attachments/assets/893288b1-f526-4bd5-9ed6-a1a7c1318ce5" />

---

## 🧩 Problem Statement

The rental market is plagued by fraudulent listings that deceive home-seekers into losing deposits and personal information. Traditional fraud detection systems rely on rigid rule-based logic or black-box ML models that flag listings without explanation — making it impossible for analysts to trust, investigate, or act on decisions.

**HomeScout solves this by combining:**
- Machine learning for accurate fraud classification
- Retrieval-Augmented Generation (RAG) for contextual historical comparison
- Large Language Model reasoning for human-readable, structured fraud explanations
- An interactive dashboard for real-time analyst decision support

The result is a shift from black-box predictions to **explainable, decision-support AI** for rental fraud analysis.

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| **Precision** | 0.9921 |
| **Recall** | 0.9363 |
| **F1 Score** | 0.9634 |
| **ROC-AUC** | 0.9853 |
| **Recall@100** | 0.3745 |
| **Accuracy** | 99.8% |
| **Frauds Caught** | 250 / 267 (93.6%) |
| **False Positives** | 2 / 9,604 (0.02%) |

> Evaluated on 9,871 held-out listings (20% test split, stratified by fraud label).

### Confusion Matrix

```
                Predicted Legit    Predicted Fraud
Actual Legit         9,602               2
Actual Fraud            17             250
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                            │
│  Renthop NYC Listings (49,352) + Rule + LLM Labels      │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     FEATURE EXTRACTION      │
          │  DistilBERT [CLS] Embeddings│
          │  (768-dim, vanilla pretrain)│
          │  + Structured Features (14) │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │    PYTORCH NEURAL NETWORK   │
          │  Linear(782 → 512)          │
          │  BatchNorm + ReLU + Drop    │
          │  Linear(512 → 256)          │
          │  BatchNorm + ReLU + Drop    │
          │  Linear(256 → 128)          │
          │  BatchNorm + ReLU + Drop    │
          │  Linear(128 → 1) + Sigmoid  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       RAG PIPELINE          │
          │  FAISS Vector Store         │
          │  (49,352 × 384-dim)         │
          │  Cosine Similarity Search   │
          │  Top-K Historical Retrieval │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │    GROQ LLM EXPLANATION     │
          │  Llama 3.3-70B-Versatile    │
          │  Structured Fraud Report    │
          │  Risk + Patterns + Rec.     │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     STREAMLIT DASHBOARD     │
          │  Real-time Fraud Scoring    │
          │  Interactive Map (Folium)   │
          │  Plotly Visualizations      │
          └─────────────────────────────┘
```

---

## 🤖 Models & Technologies

### 1. DistilBERT — Embedding Extractor
- **Model:** `distilbert-base-uncased` (HuggingFace Transformers)
- **Usage:** Extract `[CLS]` token embeddings (768-dim) from listing descriptions enriched with structured metadata
- **Why DistilBERT:** 40% smaller than BERT-base, 60% faster, retains 97% of performance — ideal for processing 49K listings on CPU
- **Input:** `"Price: $X/month. Bedrooms: N. Description: [text]"`
- **Output:** 768-dimensional dense vector per listing

### 2. PyTorch Neural Network — Fraud Classifier
- **Framework:** PyTorch 2.x
- **Architecture:**

```
Input: 782 features (768 BERT + 14 structured)
  ↓
Linear(782, 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
  ↓
Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
  ↓
Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.2)
  ↓
Linear(128, 1) → BCEWithLogitsLoss
```

- **Loss Function:** `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (97.3% legit vs 2.7% fraud)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Regularization:** BatchNorm + Dropout + gradient clipping (max_norm=1.0)
- **Early Stopping:** patience=5 on validation F1
- **Best Epoch:** 2 (early stopping triggered at epoch 7)

### 3. Sentence Transformers — RAG Embedder
- **Model:** `all-MiniLM-L6-v2`
- **Embedding Dim:** 384
- **Usage:** Embed listing summaries for FAISS vector store
- **Normalization:** L2-normalized for cosine similarity via inner product

### 4. FAISS — Vector Similarity Search
- **Index Type:** `IndexFlatIP` (exact inner product = cosine similarity with normalized vectors)
- **Vectors:** 49,352 listing summaries
- **Dimensionality:** 384
- **Retrieval:** Top-K most similar historical listings at inference time

### 5. Groq LLM — Fraud Explanation Generator
- **Model:** `llama-3.3-70b-versatile` via Groq API
- **Why Groq:** Sub-second inference latency, free tier with generous limits
- **Temperature:** 0.2 (low — consistent, factual outputs)
- **Max Tokens:** 700
- **Output Format:** 5-section structured fraud report

### 6. LoRA / PEFT — Fine-Tuning (Optional)
- **Library:** HuggingFace PEFT
- **Config:** LoRA rank=8, alpha=16, dropout=0.1
- **Target Modules:** `q_lin`, `v_lin` (DistilBERT attention)
- **Status:** Available via `models/finetune_bert.py` — skipped in baseline since vanilla BERT already achieved F1=0.9634

---

## 📊 Structured Feature Engineering

14 hand-crafted features engineered from raw listing data:

| Feature | Description |
|---|---|
| `price` | Monthly rent in USD |
| `bedrooms` / `bathrooms` | Room counts |
| `price_per_bed` | Price ÷ bedrooms (anomaly signal) |
| `scam_keyword_count` | Count of scam phrases in description |
| `has_scam_keywords` | Binary: any scam keywords present |
| `suspiciously_low_price` | Binary: price < $800/month |
| `suspiciously_high_price` | Binary: price > $15,000/month |
| `no_photos` | Binary: zero photos attached |
| `short_description` | Binary: description < 50 chars |
| `photo_count` | Number of listing photos |
| `fraud_score` | Rule-based composite score |
| `latitude` / `longitude` | Geographic coordinates |

### Fraud Score Formula
```
fraud_score = 3×has_scam_keywords + 2×suspiciously_low_price
            + no_photos + short_description
```

---

## 🏷️ Fraud Label Generation (Phase 1)

Since no labeled rental fraud dataset was available, labels were engineered using a hybrid approach:

**Step 1 — Rule Engine:**
- Score = 0 → `legit`
- Score ≥ 3 → `fraud`
- Score 1–2 → `edge_case`

**Step 2 — Groq LLM Labeling:**
- 200 edge cases sent to `llama-3.3-70b-versatile`
- LLM classifies each as fraud/legit based on full listing context
- Remaining unlabeled edge cases default to legit

**Final Distribution:**
```
Total listings : 49,352
Fraudulent     : 1,336  (2.7%)
Legitimate     : 48,016 (97.3%)
```

---

## 🔍 RAG Pipeline

The Retrieval-Augmented Generation pipeline adds contextual grounding to LLM explanations:

```
1. User submits listing
      ↓
2. Build query text from listing features
      ↓
3. Embed query with all-MiniLM-L6-v2 (384-dim)
      ↓
4. FAISS cosine similarity search → Top-K results
      ↓
5. Retrieve metadata for similar historical listings
      ↓
6. Build structured prompt:
   - Listing features
   - ML verdict + probability
   - Similar historical cases (with fraud labels)
      ↓
7. Groq LLM generates 5-section fraud report
      ↓
8. Display in Streamlit dashboard
```

### LLM Output Structure
Each analysis generates:
1. **Risk Analysis** — Why the listing is suspicious or safe
2. **Pattern Comparison** — How it compares to retrieved historical cases
3. **Key Risk Indicators** — Top 3 red or green flags
4. **Recommendation** — Suggested analyst action
5. **Executive Summary** — One-sentence verdict for the dashboard

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/homescout
cd homescout
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn torch transformers peft
pip install sentence-transformers faiss-cpu groq langchain
pip install streamlit plotly folium streamlit-folium
pip install python-dotenv joblib accelerate datasets
```

### 3. Configure API Key
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env
```

Get your free Groq API key at: https://console.groq.com

### 4. Get Dataset
Download `train.json` from the [Two Sigma Connect Rental Listing Kaggle Competition](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data) and place it in `data/`.

### 5. Run Pipeline (in order)

```bash
# Phase 1 — Generate fraud labels
python scraper/generate_labels.py

# Phase 2 — Extract BERT embeddings (baseline)
python models/extract_embeddings.py

# Phase 2 — Train PyTorch classifier
python models/train_model.py

# Phase 3 — Build FAISS vector store
python rag/build_vectorstore.py

# Phase 4 — Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
homescout/
├── app.py                        # Streamlit dashboard
├── .env                          # API keys
├── data/
│   ├── train.json                # Renthop raw data (Kaggle)
│   └── listings.csv              # Processed + labeled dataset
├── scraper/
│   └── generate_labels.py        # Rule + LLM fraud label generation
├── models/
│   ├── extract_embeddings.py     # Vanilla BERT embedding extractor
│   ├── train_model.py            # PyTorch NN fraud classifier
│   ├── finetune_bert.py          # LoRA/PEFT fine-tuning (optional)
│   ├── fraud_classifier.pt       # Saved PyTorch model weights
│   ├── bert_embeddings.npy       # Precomputed BERT embeddings
│   ├── bert_labels.npy           # Corresponding labels
│   ├── scaler.pkl                # StandardScaler for structured features
│   └── classifier_config.pkl     # Model config (dims, feature names)
├── rag/
│   ├── build_vectorstore.py      # FAISS index builder
│   └── rag_pipeline.py           # Retrieval + LLM explanation pipeline
└── vectorstores/
    ├── listings.index            # FAISS index (49,352 vectors)
    ├── listings_metadata.csv     # Listing metadata for retrieval display
    └── config.pkl                # Embedding model config
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Fraud Detection** | PyTorch NN on BERT + structured features |
| **RAG Pipeline** | FAISS top-K retrieval + Groq LLM grounding |
| **LLM Explanations** | 5-section structured analyst report |
| **Fraud Flag Chips** | Visual indicators for scam keywords, price anomalies, missing photos |
| **Fraud Probability Gauge** | Plotly gauge chart with risk zones |
| **Similar Case Cards** | Historical listings ranked by cosine similarity |
| **Interactive Map** | Folium map with listing location pin |
| **Dataset Explorer** | Price distributions, fraud score histograms, raw data viewer |
| **Model Metrics Tab** | Precision, Recall, F1, AUC, confusion matrix, architecture |
| **Session State** | Results persist across Streamlit reruns |
| **LoRA Fine-tuning** | Optional PEFT fine-tuning for improved embeddings |

---

## ⚖️ Tradeoffs & Limitations

### Tradeoffs
- **Explainability vs Latency:** LLM inference adds ~2–4 seconds per analysis but provides human-readable reasoning
- **Label Quality:** Fraud labels are engineered (rule + LLM), not human-annotated — may contain noise
- **RAG Grounding:** Retrieved cases improve LLM accuracy but add system complexity and dependency on retrieval quality
- **CPU vs GPU:** All steps run on CPU; GPU would reduce embedding extraction from ~8 min to ~45 sec

### Limitations
- LLM outputs are post-hoc reasoning — not guaranteed to be faithful to model internals
- Recall@100 (0.3745) is relatively low — top 100 predictions catch only 37% of fraud
- No real-time scraping — uses static Kaggle dataset
- LoRA fine-tuning not completed on CPU due to time constraints (1–2 hrs)

### Future Improvements
- GPU-accelerated LoRA fine-tuning to improve Recall@100
- Live scraping from Craigslist / Zillow APIs
- Image analysis (reverse image search, photoshop detection with OpenCV)
- Redis caching for repeated listing queries
- Docker containerization + cloud deployment (GCP/AWS)
- Human-annotated fraud labels for ground truth validation

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | PyTorch 2.x, HuggingFace Transformers |
| **LLM Fine-tuning** | PEFT, LoRA, Accelerate |
| **Embeddings** | DistilBERT, Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Search** | FAISS (IndexFlatIP) |
| **LLM Inference** | Groq API, Llama 3.3-70B-Versatile |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Streamlit, Plotly, Folium |
| **ML Utilities** | StandardScaler, BCEWithLogitsLoss, AdamW |
| **Storage** | CSV, NumPy (.npy), Joblib (.pkl), FAISS (.index) |

---

## 📈 Training Details

| Parameter | Value |
|---|---|
| Dataset Size | 49,352 listings |
| Train / Test Split | 80% / 20% (stratified) |
| Batch Size | 64 |
| Max Epochs | 30 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Dropout | 0.3 (blocks 1–2), 0.2 (block 3) |
| Pos Weight (fraud) | ~35.9× (legit:fraud ratio) |
| Best Epoch | 2 |
| Early Stopping | Patience 5 on val F1 |
| Gradient Clipping | max_norm = 1.0 |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- [Renthop / Two Sigma Connect](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries) for the rental listing dataset
- [Groq](https://groq.com) for ultra-fast LLM inference
- [HuggingFace](https://huggingface.co) for Transformers and PEFT libraries
- [Meta AI](https://ai.meta.com) for the Llama 3.3 model family
