"""
HomeScout – FAISS Vector Store Builder
Embeds listing summaries and stores them in a FAISS index for similarity search
"""

import os
import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim
BATCH_SIZE      = 64


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_data(path="data/listings.csv"):
    print("📂 Loading listings.csv...")
    df = pd.read_csv(path)
    df["description"] = df["description"].fillna("")
    print(f"✅ Loaded {len(df)} listings")
    return df


# ──────────────────────────────────────────────
# 2. BUILD LISTING SUMMARIES
# ──────────────────────────────────────────────

def build_summaries(df):
    """Convert each listing into a rich text summary for embedding."""
    print("📝 Building listing summaries...")

    def summarize(row):
        fraud_status = "FRAUDULENT" if row["is_fraud"] == 1 else "LEGITIMATE"
        flags = []
        if row.get("has_scam_keywords", 0): flags.append("scam keywords detected")
        if row.get("suspiciously_low_price", 0): flags.append("suspiciously low price")
        if row.get("no_photos", 0): flags.append("no photos")
        if row.get("short_description", 0): flags.append("very short description")
        flag_str = ", ".join(flags) if flags else "no fraud flags"

        return (
            f"Listing status: {fraud_status}. "
            f"Price: ${row['price']}/month. "
            f"Bedrooms: {row['bedrooms']}, Bathrooms: {row['bathrooms']}. "
            f"Price per bedroom: ${row.get('price_per_bed', 0):.0f}. "
            f"Photos: {row.get('photo_count', 0)}. "
            f"Fraud flags: {flag_str}. "
            f"Description: {str(row['description'])[:150]}"
        )

    df["summary"] = df.apply(summarize, axis=1)
    print(f"✅ Built {len(df)} summaries")
    return df


# ──────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS
# ──────────────────────────────────────────────

def generate_embeddings(summaries):
    print(f"\n🤖 Embedding summaries with {EMBEDDING_MODEL}...")
    model      = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        summaries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True   # cosine similarity friendly
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return model, embeddings


# ──────────────────────────────────────────────
# 4. BUILD FAISS INDEX
# ──────────────────────────────────────────────

def build_faiss_index(embeddings):
    print("\n🗄️  Building FAISS index...")
    dim   = embeddings.shape[1]

    # Inner product index (works with normalized embeddings = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"✅ FAISS index built — {index.ntotal} vectors, {dim} dimensions")
    return index


# ──────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────

def save(index, df, embedding_model):
    os.makedirs("vectorstores", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, "vectorstores/listings.index")

    # Save metadata (summaries + labels for retrieval display)
    metadata = df[["summary", "is_fraud", "price", "bedrooms",
                   "bathrooms", "photo_count", "fraud_score",
                   "display_address", "description"]].copy()
    metadata.to_csv("vectorstores/listings_metadata.csv", index=False)

    # Save embedding model name for later use
    joblib.dump({"model_name": EMBEDDING_MODEL}, "vectorstores/config.pkl")

    print("\n💾 Saved:")
    print("   vectorstores/listings.index")
    print("   vectorstores/listings_metadata.csv")
    print("   vectorstores/config.pkl")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df                = load_data("data/listings.csv")
    df                = build_summaries(df)
    model, embeddings = generate_embeddings(df["summary"].tolist())
    index             = build_faiss_index(embeddings)
    save(index, df, model)

    print("\n🎉 Vector store ready! Run rag_pipeline.py next.")