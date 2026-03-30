"""
HomeScout – RAG Pipeline
Retrieves similar historical listings via FAISS + generates fraud explanation via Groq
"""

import os
import pandas as pd
import numpy as np
import faiss
import joblib
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ──────────────────────────────────────────────
# 1. LOAD VECTOR STORE
# ──────────────────────────────────────────────

def load_vectorstore():
    print("🗄️  Loading FAISS vector store...")
    index    = faiss.read_index("vectorstores/listings.index")
    metadata = pd.read_csv("vectorstores/listings_metadata.csv")
    config   = joblib.load("vectorstores/config.pkl")
    model    = SentenceTransformer(config["model_name"])
    print(f"✅ Loaded {index.ntotal} vectors")
    return index, metadata, model


# ──────────────────────────────────────────────
# 2. RETRIEVE SIMILAR LISTINGS
# ──────────────────────────────────────────────

def retrieve_similar(query_text, index, metadata, embed_model, top_k=5):
    query_embedding = embed_model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = metadata.iloc[idx]
        results.append({
            "similarity":  float(score),
            "is_fraud":    int(row["is_fraud"]),
            "price":       row["price"],
            "bedrooms":    row["bedrooms"],
            "bathrooms":   row["bathrooms"],
            "photo_count": row["photo_count"],
            "fraud_score": row["fraud_score"],
            "summary":     row["summary"],
            "description": str(row["description"])[:200]
        })
    return results


# ──────────────────────────────────────────────
# 3. BUILD PROMPT
# ──────────────────────────────────────────────

def build_prompt(listing, ml_prediction, ml_probability, similar_listings):
    fraud_label = "FRAUD" if ml_prediction == 1 else "LEGITIMATE"

    similar_cases_text = ""
    for i, case in enumerate(similar_listings, 1):
        status = "FRAUD" if case["is_fraud"] == 1 else "LEGIT"
        similar_cases_text += (
            f"\nCase {i} ({status}, similarity: {case['similarity']:.2f}):\n"
            f"  Price: ${case['price']}/month | "
            f"Beds: {case['bedrooms']} | "
            f"Photos: {case['photo_count']} | "
            f"Fraud score: {case['fraud_score']}\n"
            f"  Summary: {case['summary'][:200]}\n"
        )

    return f"""You are a rental fraud detection expert. Analyze the following listing and provide a structured fraud assessment.

## LISTING UNDER REVIEW
- Price: ${listing.get('price', 'N/A')}/month
- Bedrooms: {listing.get('bedrooms', 'N/A')}
- Bathrooms: {listing.get('bathrooms', 'N/A')}
- Photos: {listing.get('photo_count', 0)}
- Scam keywords detected: {listing.get('scam_keyword_count', 0)}
- Price suspiciously low: {bool(listing.get('suspiciously_low_price', 0))}
- Description: {str(listing.get('description', ''))[:300]}

## ML MODEL PREDICTION
- Verdict: {fraud_label}
- Fraud probability: {ml_probability:.1%}

## SIMILAR HISTORICAL CASES (retrieved via semantic search)
{similar_cases_text}

## YOUR TASK
Based on the listing details, ML prediction, and similar historical cases above, provide:

1. **Risk Analysis**: Why is this listing suspicious or safe?
2. **Pattern Comparison**: How does it compare to retrieved historical cases?
3. **Key Risk Indicators**: List the top 3 red flags (or green flags if legitimate)
4. **Recommendation**: What should the analyst do next?
5. **Executive Summary**: One sentence verdict for the dashboard

Be specific, concise, and base your analysis ONLY on the provided information."""


# ──────────────────────────────────────────────
# 4. GROQ LLM EXPLANATION
# ──────────────────────────────────────────────

def generate_explanation(prompt):
    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# 5. FULL RAG PIPELINE
# ──────────────────────────────────────────────

def analyze_listing(listing: dict, ml_prediction: int, ml_probability: float,
                    index, metadata, embed_model, top_k=5):
    query = (
        f"Price: ${listing.get('price', 0)}/month. "
        f"Bedrooms: {listing.get('bedrooms', 0)}, "
        f"Bathrooms: {listing.get('bathrooms', 0)}. "
        f"Photos: {listing.get('photo_count', 0)}. "
        f"Description: {str(listing.get('description', ''))[:150]}"
    )

    similar    = retrieve_similar(query, index, metadata, embed_model, top_k=top_k)
    prompt     = build_prompt(listing, ml_prediction, ml_probability, similar)
    explanation = generate_explanation(prompt)

    return {
        "ml_prediction":  ml_prediction,
        "ml_probability": ml_probability,
        "similar_cases":  similar,
        "explanation":    explanation
    }


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    index, metadata, embed_model = load_vectorstore()

    test_listing = {
        "price":                  800,
        "bedrooms":               3,
        "bathrooms":              2,
        "photo_count":            0,
        "scam_keyword_count":     2,
        "suspiciously_low_price": 1,
        "description": (
            "Beautiful 3BR apartment available immediately. "
            "Send deposit via wire transfer. God bless. No credit check needed. "
            "Contact via WhatsApp only. Overseas owner."
        )
    }

    print("\n🔍 Analyzing test listing...")
    print("=" * 60)

    result = analyze_listing(
        listing=test_listing,
        ml_prediction=1,
        ml_probability=0.94,
        index=index,
        metadata=metadata,
        embed_model=embed_model
    )

    print(f"🤖 ML Prediction : {'FRAUD' if result['ml_prediction'] == 1 else 'LEGIT'}")
    print(f"📊 Probability   : {result['ml_probability']:.1%}")
    print(f"\n📋 Similar Cases Retrieved: {len(result['similar_cases'])}")
    for i, case in enumerate(result['similar_cases'], 1):
        status = "🚨 FRAUD" if case['is_fraud'] == 1 else "✅ LEGIT"
        print(f"  {i}. {status} | ${case['price']}/mo | sim: {case['similarity']:.3f}")

    print(f"\n💬 LLM Explanation:\n")
    print(result["explanation"])
    print("\n🎉 RAG Pipeline working! Ready for Streamlit UI.")