"""
HomeScout – Fraud Label Generator
Uses rule-based logic + Groq LLM to label Renthop listings as fraudulent or legitimate.
"""

import json
import pandas as pd
import numpy as np
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_renthop(path="data/train.json"):
    print("📂 Loading Renthop dataset...")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} listings")
    return df


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

SCAM_KEYWORDS = [
    "wire transfer", "western union", "money order", "god bless",
    "overseas", "abroad", "missionary", "deployed", "military",
    "send deposit", "cashier check", "no credit check", "guaranteed approval",
    "too good to be true", "urgent", "act fast", "limited time",
    "whatsapp only", "email only", "no viewing"
]

def engineer_features(df):
    print("⚙️  Engineering fraud features...")

    # Price per bedroom
    df["bedrooms"] = df["bedrooms"].replace(0, 1)  # avoid div by zero
    df["price_per_bed"] = df["price"] / df["bedrooms"]

    # Price anomaly flag — below 10th percentile per neighborhood
    price_threshold = df.groupby("building_id")["price"].transform(
        lambda x: x.quantile(0.10)
    )
    df["suspiciously_low_price"] = (df["price"] < price_threshold * 0.6).astype(int)

    # Global price anomaly fallback
    global_low = df["price"].quantile(0.05)
    df["suspiciously_low_price"] = df["suspiciously_low_price"] | (df["price"] < global_low)

    # Scam keyword detection in description
    df["description"] = df["description"].fillna("").str.lower()
    df["scam_keyword_count"] = df["description"].apply(
        lambda text: sum(kw in text for kw in SCAM_KEYWORDS)
    )
    df["has_scam_keywords"] = (df["scam_keyword_count"] > 0).astype(int)

    # Missing critical fields
    df["missing_street"] = df["street_address"].apply(
        lambda x: 1 if pd.isna(x) or str(x).strip() == "" else 0
    ) if "street_address" in df.columns else 0

    df["missing_display_address"] = df["display_address"].apply(
        lambda x: 1 if pd.isna(x) or str(x).strip() == "" else 0
    ) if "display_address" in df.columns else 0

    # Few or no photos
    df["photo_count"] = df["photos"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["no_photos"] = (df["photo_count"] == 0).astype(int)

    # Very short description
    df["desc_length"] = df["description"].apply(len)
    df["short_description"] = (df["desc_length"] < 50).astype(int)

    # Abnormally high price (luxury fraud bait)
    global_high = df["price"].quantile(0.98)
    df["suspiciously_high_price"] = (df["price"] > global_high).astype(int)

    print("✅ Feature engineering complete")
    return df


# ──────────────────────────────────────────────
# 3. RULE-BASED FRAUD SCORING
# ──────────────────────────────────────────────

def rule_based_fraud_score(row):
    score = 0
    score += 3 * row.get("has_scam_keywords", 0)
    score += 2 * row.get("suspiciously_low_price", 0)
    score += 1 * row.get("no_photos", 0)
    score += 1 * row.get("short_description", 0)
    score += 1 * row.get("missing_display_address", 0)
    score += 1 * row.get("suspiciously_high_price", 0)
    return score

def apply_rule_labels(df):
    print("🔍 Applying rule-based fraud scoring...")
    df["fraud_score"] = df.apply(rule_based_fraud_score, axis=1)

    # Clear labels: score >= 3 → fraud, score 0 → legit, in between → edge case
    df["rule_label"] = df["fraud_score"].apply(
        lambda s: "fraud" if s >= 3 else ("legit" if s == 0 else "edge_case")
    )

    fraud_count = (df["rule_label"] == "fraud").sum()
    legit_count = (df["rule_label"] == "legit").sum()
    edge_count  = (df["rule_label"] == "edge_case").sum()

    print(f"  🚨 Fraud:     {fraud_count}")
    print(f"  ✅ Legit:     {legit_count}")
    print(f"  ❓ Edge case: {edge_count}")
    return df


# ──────────────────────────────────────────────
# 4. GROQ LLM LABELING FOR EDGE CASES
# ──────────────────────────────────────────────

def build_prompt(row):
    return f"""You are a rental fraud detection expert. Analyze this listing and respond with ONLY one word: "fraud" or "legit".

Listing Details:
- Price: ${row.get('price', 'N/A')}/month
- Bedrooms: {row.get('bedrooms', 'N/A')}
- Bathrooms: {row.get('bathrooms', 'N/A')}
- Location: {row.get('display_address', 'N/A')}, {row.get('city', 'N/A') if 'city' in row else 'NYC'}
- Description: {str(row.get('description', ''))[:300]}
- Photos available: {row.get('photo_count', 0)}
- Price per bedroom: ${row.get('price_per_bed', 0):.0f}
- Scam keywords found: {row.get('scam_keyword_count', 0)}

Respond with ONLY one word: fraud or legit"""


def llm_label_edge_cases(df, max_llm_calls=200):
    """Use Groq to label edge cases — capped to avoid rate limits."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    edge_df = df[df["rule_label"] == "edge_case"].copy()

    if len(edge_df) == 0:
        print("ℹ️  No edge cases to label with LLM")
        return df

    # Cap LLM calls
    sample = edge_df.sample(min(max_llm_calls, len(edge_df)), random_state=42)
    print(f"🤖 Sending {len(sample)} edge cases to Groq LLM...")

    llm_labels = {}
    for i, (idx, row) in enumerate(sample.iterrows()):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(row)}],
                max_tokens=5,
                temperature=0.0
            )
            label = response.choices[0].message.content.strip().lower()
            label = "fraud" if "fraud" in label else "legit"
            llm_labels[idx] = label

            if (i + 1) % 20 == 0:
                print(f"  ✅ Labeled {i+1}/{len(sample)}")
                time.sleep(1)  # rate limit buffer

        except Exception as e:
            print(f"  ⚠️  Error on listing {idx}: {e}")
            llm_labels[idx] = "legit"  # default to legit on error
            time.sleep(2)

    # Apply LLM labels to edge cases
    df["llm_label"] = None
    for idx, label in llm_labels.items():
        df.at[idx, "llm_label"] = label

    # Remaining edge cases without LLM label → default legit
    remaining = df[(df["rule_label"] == "edge_case") & (df["llm_label"].isna())].index
    df.loc[remaining, "llm_label"] = "legit"

    print(f"✅ LLM labeling complete")
    return df


# ──────────────────────────────────────────────
# 5. COMBINE LABELS → FINAL is_fraud COLUMN
# ──────────────────────────────────────────────

def combine_labels(df):
    def final_label(row):
        if row["rule_label"] == "fraud":
            return 1
        elif row["rule_label"] == "legit":
            return 0
        else:
            # edge case — use LLM label
            return 1 if row.get("llm_label") == "fraud" else 0

    df["is_fraud"] = df.apply(final_label, axis=1)

    total   = len(df)
    fraud   = df["is_fraud"].sum()
    legit   = total - fraud
    print(f"\n📊 Final Label Distribution:")
    print(f"  Total listings : {total}")
    print(f"  Fraudulent     : {fraud}  ({fraud/total*100:.1f}%)")
    print(f"  Legitimate     : {legit} ({legit/total*100:.1f}%)")
    return df


# ──────────────────────────────────────────────
# 6. SAVE CLEAN DATASET
# ──────────────────────────────────────────────

KEEP_COLS = [
    "listing_id", "price", "bedrooms", "bathrooms",
    "display_address", "latitude", "longitude",
    "description", "photo_count", "price_per_bed",
    "scam_keyword_count", "has_scam_keywords",
    "suspiciously_low_price", "suspiciously_high_price",
    "no_photos", "short_description",
    "fraud_score", "rule_label", "llm_label", "is_fraud"
]

def save_dataset(df, out_path="data/listings.csv"):
    os.makedirs("data", exist_ok=True)
    cols = [c for c in KEEP_COLS if c in df.columns]
    df[cols].to_csv(out_path, index=False)
    print(f"\n💾 Saved to {out_path} ({len(df)} rows, {len(cols)} columns)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_renthop("data/train.json")
    df = engineer_features(df)
    df = apply_rule_labels(df)
    df = llm_label_edge_cases(df, max_llm_calls=200)
    df = combine_labels(df)
    save_dataset(df, "data/listings.csv")
    print("\n🎉 Phase 1 complete! listings.csv is ready for Phase 2 (ML model)")