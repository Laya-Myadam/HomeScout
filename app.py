"""
HomeScout – AI-Powered Rental Fraud Detection Dashboard
Clean white professional Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="HomeScout – Fraud Detection",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Clean White Professional UI
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif;
    color: #0F0F0F;
}

[data-testid="stSidebar"] {
    background-color: #F7F7F5 !important;
    border-right: 1px solid #E8E8E4;
}

[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #E8E8E4;
}

/* ── Hide Streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif; font-weight: 700; color: #0F0F0F; }

/* ── Hero Header ── */
.hero {
    background: #0F0F0F;
    color: #FFFFFF;
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon { font-size: 3rem; }
.hero-title { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-sub { font-size: 0.95rem; color: #A0A09A; margin: 0.25rem 0 0; font-weight: 400; }

/* ── Metric Cards ── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: box-shadow 0.2s ease;
}
.metric-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
.metric-value { font-size: 2rem; font-weight: 700; color: #0F0F0F; font-family: 'DM Mono', monospace; }
.metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }

/* ── Verdict Badge ── */
.verdict-fraud {
    background: #FFF0F0;
    border: 2px solid #E53E3E;
    color: #E53E3E;
    padding: 1rem 2rem;
    border-radius: 12px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.05em;
}
.verdict-legit {
    background: #F0FFF4;
    border: 2px solid #38A169;
    color: #38A169;
    padding: 1rem 2rem;
    border-radius: 12px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.05em;
}

/* ── Section Headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E8E8E4;
}

/* ── Similar Case Cards ── */
.case-card {
    background: #FAFAF8;
    border: 1px solid #E8E8E4;
    border-left: 4px solid #E8E8E4;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
}
.case-card.fraud { border-left-color: #E53E3E; }
.case-card.legit { border-left-color: #38A169; }
.case-tag-fraud { background: #FFF0F0; color: #E53E3E; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.case-tag-legit { background: #F0FFF4; color: #38A169; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }

/* ── LLM Explanation Box ── */
.explanation-box {
    background: #FAFAF8;
    border: 1px solid #E8E8E4;
    border-radius: 12px;
    padding: 1.75rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #2D2D2D;
}

/* ── Input Styling ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1px solid #E8E8E4 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #FFFFFF !important;
    color: #0F0F0F !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #0F0F0F !important;
    box-shadow: 0 0 0 2px rgba(15,15,15,0.1) !important;
}

/* ── Button ── */
.stButton > button {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: #0F0F0F !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #E8E8E4;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500;
    color: #888 !important;
    padding: 0.6rem 1.5rem;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #0F0F0F !important;
    border-bottom: 2px solid #0F0F0F;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid #E8E8E4; margin: 1.5rem 0; }

/* ── Flag chips ── */
.flag-chip {
    display: inline-block;
    background: #FFF0F0;
    color: #E53E3E;
    border: 1px solid #FED7D7;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}
.safe-chip {
    display: inline-block;
    background: #F0FFF4;
    color: #38A169;
    border: 1px solid #C6F6D5;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# NEURAL NETWORK (must match train_model.py)
# ──────────────────────────────────────────────

class FraudClassifier(nn.Module):
    def __init__(self, structured_dim, embedding_dim, dropout=0.3):
        super().__init__()
        input_dim = structured_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, structured, embeddings):
        x = torch.cat([structured, embeddings], dim=1)
        return self.net(x).squeeze(1)

# ──────────────────────────────────────────────
# LOAD MODELS (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    config     = joblib.load("models/classifier_config.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    model      = FraudClassifier(config["structured_dim"], config["embedding_dim"])
    model.load_state_dict(torch.load("models/fraud_classifier.pt", map_location="cpu"))
    model.eval()
    return model, scaler, config

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert      = AutoModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    return tokenizer, bert

@st.cache_resource
def load_vectorstore():
    index    = faiss.read_index("vectorstores/listings.index")
    metadata = pd.read_csv("vectorstores/listings_metadata.csv")
    cfg      = joblib.load("vectorstores/config.pkl")
    embedder = SentenceTransformer(cfg["model_name"])
    return index, metadata, embedder

# ──────────────────────────────────────────────
# INFERENCE HELPERS
# ──────────────────────────────────────────────

STRUCTURED_FEATURES = [
    "price", "bedrooms", "bathrooms", "price_per_bed",
    "scam_keyword_count", "has_scam_keywords",
    "suspiciously_low_price", "suspiciously_high_price",
    "no_photos", "short_description", "photo_count",
    "fraud_score", "latitude", "longitude"
]

SCAM_KEYWORDS = [
    "wire transfer", "western union", "money order", "god bless",
    "overseas", "abroad", "missionary", "deployed", "military",
    "send deposit", "cashier check", "no credit check", "guaranteed approval",
    "urgent", "act fast", "whatsapp only", "email only", "no viewing"
]

def engineer_input(listing: dict) -> dict:
    desc  = listing.get("description", "").lower()
    price = listing.get("price", 0)
    beds  = max(listing.get("bedrooms", 1), 1)
    photos = listing.get("photo_count", 0)

    scam_count = sum(kw in desc for kw in SCAM_KEYWORDS)
    listing["price_per_bed"]            = price / beds
    listing["scam_keyword_count"]       = scam_count
    listing["has_scam_keywords"]        = int(scam_count > 0)
    listing["suspiciously_low_price"]   = int(price < 800)
    listing["suspiciously_high_price"]  = int(price > 15000)
    listing["no_photos"]                = int(photos == 0)
    listing["short_description"]        = int(len(desc) < 50)
    listing["fraud_score"]              = (
        3 * listing["has_scam_keywords"] +
        2 * listing["suspiciously_low_price"] +
        listing["no_photos"] +
        listing["short_description"]
    )
    listing.setdefault("latitude",  40.7128)
    listing.setdefault("longitude", -74.0060)
    return listing

def get_bert_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors="pt", max_length=128,
                       padding="max_length", truncation=True)
    with torch.no_grad():
        out = bert_model(**inputs)
    return out.last_hidden_state[:, 0, :].numpy()

def predict(listing, model, scaler, config, tokenizer, bert_model):
    listing = engineer_input(listing)
    struct  = np.array([[listing.get(f, 0) for f in STRUCTURED_FEATURES]], dtype=float)
    struct  = scaler.transform(struct)

    text = (f"Price: ${listing['price']}/month. "
            f"Bedrooms: {listing['bedrooms']}. "
            f"Description: {listing.get('description','')[:200]}")
    emb = get_bert_embedding(text, tokenizer, bert_model)

    s_t = torch.tensor(struct, dtype=torch.float32)
    e_t = torch.tensor(emb,    dtype=torch.float32)

    with torch.no_grad():
        logit = model(s_t, e_t)
        prob  = torch.sigmoid(logit).item()

    return int(prob > 0.5), prob, listing

def retrieve_similar(query, index, metadata, embedder, top_k=5):
    qe = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(qe, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        row = metadata.iloc[idx]
        results.append({
            "similarity": float(score),
            "is_fraud":   int(row["is_fraud"]),
            "price":      row["price"],
            "bedrooms":   row["bedrooms"],
            "photo_count":row["photo_count"],
            "fraud_score":row["fraud_score"],
            "summary":    row["summary"]
        })
    return results

def get_llm_explanation(listing, prediction, probability, similar_cases):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    fraud_label = "FRAUD" if prediction == 1 else "LEGITIMATE"

    cases_text = ""
    for i, c in enumerate(similar_cases, 1):
        status = "FRAUD" if c["is_fraud"] == 1 else "LEGIT"
        cases_text += (f"\nCase {i} ({status}, sim: {c['similarity']:.2f}): "
                       f"${c['price']}/mo | Beds: {c['bedrooms']} | "
                       f"Photos: {c['photo_count']} | Score: {c['fraud_score']}\n")

    prompt = f"""You are a rental fraud detection expert. Analyze this listing concisely.

LISTING: Price ${listing.get('price')}/mo | Beds: {listing.get('bedrooms')} | 
Photos: {listing.get('photo_count',0)} | Scam keywords: {listing.get('scam_keyword_count',0)}
Description: {str(listing.get('description',''))[:250]}

ML VERDICT: {fraud_label} ({probability:.1%} fraud probability)

SIMILAR HISTORICAL CASES:{cases_text}

Provide a structured analysis with these 5 sections:
1. **Risk Analysis** 
2. **Pattern Comparison**
3. **Key Risk Indicators** (top 3)
4. **Recommendation**
5. **Executive Summary** (one sentence)

Be concise and base analysis ONLY on provided data."""

    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700, temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ──────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────

def gauge_chart(probability):
    color = "#E53E3E" if probability > 0.5 else "#38A169"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "family": "DM Mono", "color": "#0F0F0F"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11, "color": "#888"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#F7F7F5",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "#F0FFF4"},
                {"range": [40, 65],  "color": "#FFFBEB"},
                {"range": [65, 100], "color": "#FFF0F0"}
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "DM Sans"}
    )
    return fig

def similarity_bar_chart(similar_cases):
    labels = [f"Case {i+1} ({'🚨' if c['is_fraud'] else '✅'})" for i, c in enumerate(similar_cases)]
    colors = ["#E53E3E" if c["is_fraud"] else "#38A169" for c in similar_cases]
    sims   = [c["similarity"] for c in similar_cases]

    fig = go.Figure(go.Bar(
        x=sims, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{s:.3f}" for s in sims], textposition="outside",
        textfont={"family": "DM Mono", "size": 11}
    ))
    fig.update_layout(
        height=220, margin=dict(t=10, b=10, l=10, r=50),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(range=[0.7, 1.0], showgrid=True, gridcolor="#F0F0EE",
                   title="Cosine Similarity", titlefont={"size": 11}),
        yaxis=dict(showgrid=False),
        font={"family": "DM Sans", "size": 12}
    )
    return fig

def map_listing(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=14,
                   tiles="CartoDB positron")
    folium.Marker(
        [lat, lon],
        popup="📍 Listing Location",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)
    return m

# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🏠</div>
        <div>
            <div class="hero-title">HomeScout</div>
            <div class="hero-sub">AI-Powered Rental Fraud Detection & Neighborhood Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        try:
            clf_model, scaler, config = load_models()
            tokenizer, bert_model     = load_bert()
            index, metadata, embedder = load_vectorstore()
            models_loaded = True
        except Exception as e:
            st.error(f"⚠️ Error loading models: {e}")
            st.info("Make sure you've run: extract_embeddings.py → train_model.py → build_vectorstore.py")
            models_loaded = False

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍  Analyze Listing", "📊  Model Metrics", "📁  Dataset Explorer"])

    # ── TAB 1: Analyze Listing ──
    with tab1:
        col_input, col_results = st.columns([1, 1.6], gap="large")

        with col_input:
            st.markdown('<div class="section-header">Listing Details</div>', unsafe_allow_html=True)

            price    = st.number_input("Monthly Rent ($)", min_value=100, max_value=50000, value=1200, step=50)
            col1, col2 = st.columns(2)
            with col1:
                bedrooms  = st.number_input("Bedrooms",  min_value=0, max_value=10, value=2)
            with col2:
                bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)

            col3, col4 = st.columns(2)
            with col3:
                photos = st.number_input("Photo Count", min_value=0, max_value=50, value=5)
            with col4:
                top_k = st.selectbox("Similar Cases", [3, 5, 7, 10], index=1)

            st.markdown('<div class="section-header">Description</div>', unsafe_allow_html=True)
            description = st.text_area(
                "Listing Description",
                value="Spacious 2BR apartment in the heart of Manhattan. "
                      "Recently renovated kitchen and bathroom. "
                      "Great natural light, hardwood floors throughout. "
                      "Close to subway and local amenities.",
                height=130,
                label_visibility="collapsed"
            )

            st.markdown('<div class="section-header">Location (optional)</div>', unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                lat = st.number_input("Latitude",  value=40.7128, format="%.4f")
            with col6:
                lon = st.number_input("Longitude", value=-74.0060, format="%.4f")

            analyze_btn = st.button("🔍  Analyze Listing", use_container_width=True)

        with col_results:
            if analyze_btn and models_loaded:
                listing = {
                    "price": price, "bedrooms": bedrooms,
                    "bathrooms": bathrooms, "photo_count": photos,
                    "description": description,
                    "latitude": lat, "longitude": lon
                }

                with st.spinner("Analyzing listing..."):
                    prediction, probability, enriched = predict(
                        listing, clf_model, scaler, config, tokenizer, bert_model
                    )

                # Verdict
                verdict_class = "verdict-fraud" if prediction == 1 else "verdict-legit"
                verdict_text  = "🚨 FRAUDULENT LISTING" if prediction == 1 else "✅ LEGITIMATE LISTING"
                st.markdown(f'<div class="{verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
                st.markdown("")

                # Gauge + flags
                gcol, fcol = st.columns([1.2, 1])
                with gcol:
                    st.plotly_chart(gauge_chart(probability), use_container_width=True, config={"displayModeBar": False})
                with fcol:
                    st.markdown('<div class="section-header">Fraud Flags</div>', unsafe_allow_html=True)
                    flags = []
                    if enriched.get("has_scam_keywords"):    flags.append("Scam keywords")
                    if enriched.get("suspiciously_low_price"): flags.append("Low price")
                    if enriched.get("no_photos"):            flags.append("No photos")
                    if enriched.get("short_description"):    flags.append("Short desc")

                    if flags:
                        for f in flags:
                            st.markdown(f'<span class="flag-chip">⚠ {f}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="safe-chip">✓ No flags detected</span>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <br>
                    <div style="font-size:0.8rem; color:#888;">
                        Fraud Score: <b style="color:#0F0F0F; font-family:'DM Mono'">{enriched.get('fraud_score', 0)}</b><br>
                        Keywords: <b style="color:#0F0F0F; font-family:'DM Mono'">{enriched.get('scam_keyword_count', 0)}</b>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="section-header">Similar Historical Cases</div>', unsafe_allow_html=True)

                with st.spinner("Retrieving similar cases..."):
                    query   = f"Price: ${price}/month. Beds: {bedrooms}. Description: {description[:150]}"
                    similar = retrieve_similar(query, index, metadata, embedder, top_k=top_k)

                # Similarity chart
                st.plotly_chart(similarity_bar_chart(similar), use_container_width=True, config={"displayModeBar": False})

                # Case cards
                for c in similar:
                    card_class = "fraud" if c["is_fraud"] else "legit"
                    tag_class  = "case-tag-fraud" if c["is_fraud"] else "case-tag-legit"
                    tag_text   = "FRAUD" if c["is_fraud"] else "LEGIT"
                    st.markdown(f"""
                    <div class="case-card {card_class}">
                        <span class="{tag_class}">{tag_text}</span>
                        <span style="margin-left:8px; font-weight:600;">${c['price']}/mo</span>
                        <span style="color:#888; font-size:0.8rem; margin-left:8px;">
                            {c['bedrooms']} bed · {c['photo_count']} photos · sim: {c['similarity']:.3f}
                        </span>
                        <div style="color:#555; font-size:0.8rem; margin-top:4px;">{c['summary'][:120]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

                # LLM Explanation
                st.markdown('<div class="section-header">AI Fraud Analysis (Groq LLM)</div>', unsafe_allow_html=True)
                with st.spinner("Generating AI explanation..."):
                    explanation = get_llm_explanation(enriched, prediction, probability, similar)
                st.markdown(f'<div class="explanation-box">{explanation.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)

                # Map
                st.markdown('<div class="section-header">Listing Location</div>', unsafe_allow_html=True)
                st_folium(map_listing(lat, lon), width="100%", height=280)

            elif not analyze_btn:
                st.markdown("""
                <div style="text-align:center; padding:4rem 2rem; color:#BBB;">
                    <div style="font-size:3rem;">🏠</div>
                    <div style="font-size:1rem; margin-top:1rem; font-weight:500;">
                        Enter listing details and click Analyze
                    </div>
                    <div style="font-size:0.85rem; margin-top:0.5rem;">
                        The AI will detect fraud signals and generate a detailed report
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: Model Metrics ──
    with tab2:
        st.markdown('<div class="section-header">Baseline Model Performance</div>', unsafe_allow_html=True)

        metrics = [
            ("Precision",  "0.9921", "Fraction of flagged listings that are truly fraudulent"),
            ("Recall",     "0.9363", "Fraction of actual fraud cases caught"),
            ("F1 Score",   "0.9634", "Harmonic mean of precision and recall"),
            ("ROC-AUC",    "0.9853", "Overall discrimination ability"),
            ("Recall@100", "0.3745", "Fraud recall in top 100 predictions"),
            ("Accuracy",   "99.8%",  "Overall classification accuracy"),
        ]

        cols = st.columns(3)
        for i, (label, val, desc) in enumerate(metrics):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.75rem; color:#AAA; margin-top:0.5rem;">{desc}</div>
                </div><br>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = [[9602, 2], [17, 250]]
        fig_cm  = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legit", "Fraud"], y=["Legit", "Fraud"],
            color_continuous_scale=[[0, "#F7F7F5"], [1, "#0F0F0F"]],
            text_auto=True
        )
        fig_cm.update_layout(
            height=320, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font={"family": "DM Sans"}, margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_showscale=False
        )
        fig_cm.update_traces(textfont={"size": 18, "family": "DM Mono"})
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        arch_cols = st.columns(3)
        arch_items = [
            ("🧠", "PyTorch Neural Net", "512→256→128→1\nBatchNorm + Dropout\nBCEWithLogitsLoss"),
            ("🤖", "DistilBERT Embeddings", "768-dim [CLS] token\nall-MiniLM-L6-v2\nVanilla pretrained"),
            ("🗄️", "FAISS Vector Store", "49,352 vectors\n384-dim embeddings\nCosine similarity"),
        ]
        for col, (icon, title, desc) in zip(arch_cols, arch_items):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-weight:700; margin:0.5rem 0 0.25rem;">{title}</div>
                    <div style="font-size:0.78rem; color:#888; white-space:pre-line;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Dataset Explorer ──
    with tab3:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        try:
            df = pd.read_csv("data/listings.csv")

            # Stats
            dcols = st.columns(4)
            stats = [
                ("Total Listings", f"{len(df):,}"),
                ("Fraudulent",     f"{df['is_fraud'].sum():,}"),
                ("Fraud Rate",     f"{df['is_fraud'].mean()*100:.1f}%"),
                ("Avg Price",      f"${df['price'].mean():,.0f}")
            ]
            for col, (label, val) in zip(dcols, stats):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Price distribution
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown('<div class="section-header">Price Distribution by Label</div>', unsafe_allow_html=True)
                fig_box = px.box(
                    df[df["price"] < 10000], x="is_fraud", y="price",
                    color="is_fraud",
                    color_discrete_map={0: "#38A169", 1: "#E53E3E"},
                    labels={"is_fraud": "Is Fraud", "price": "Price ($/mo)"}
                )
                fig_box.update_layout(
                    height=300, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    font={"family": "DM Sans"}, showlegend=False,
                    margin=dict(t=10, b=20), xaxis_title="0 = Legit, 1 = Fraud"
                )
                st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

            with pcol2:
                st.markdown('<div class="section-header">Fraud Score Distribution</div>', unsafe_allow_html=True)
                fig_hist = px.histogram(
                    df, x="fraud_score", color="is_fraud",
                    color_discrete_map={0: "#38A169", 1: "#E53E3E"},
                    barmode="overlay", opacity=0.75,
                    labels={"fraud_score": "Fraud Score", "is_fraud": "Is Fraud"}
                )
                fig_hist.update_layout(
                    height=300, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    font={"family": "DM Sans"}, margin=dict(t=10, b=20),
                    legend=dict(title="Label", bgcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            # Sample data
            st.markdown('<div class="section-header">Sample Listings</div>', unsafe_allow_html=True)
            show_fraud = st.checkbox("Show fraudulent listings only", value=False)
            sample_df  = df[df["is_fraud"] == 1] if show_fraud else df
            display_cols = ["price", "bedrooms", "bathrooms", "photo_count",
                            "fraud_score", "has_scam_keywords", "is_fraud", "description"]
            st.dataframe(
                sample_df[display_cols].head(50),
                use_container_width=True,
                height=350
            )

        except FileNotFoundError:
            st.info("Run generate_labels.py first to create the dataset.")


if __name__ == "__main__":
    main()