import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from app.model_loader import model, product_embeddings, products_df


# -------------------------------------------------
# Normalize Words (handles plural + mens/womens)
# -------------------------------------------------
def normalize(word):
    word = word.lower().strip()

    # convert mens -> men
    if word == "mens":
        return "men"
    if word == "womens":
        return "women"

    # remove plural 's'
    if word.endswith("s") and len(word) > 3:
        word = word[:-1]

    return word


# -------------------------------------------------
# Main Search Function
# -------------------------------------------------
def search_products(query, top_k=10):

    query_lower = query.lower()

    # -----------------------------
    # 1️⃣ Extract Price Filter
    # -----------------------------
    price_limit = None
    price_match = re.search(r"(under|below)\s+(\d+)", query_lower)
    if price_match:
        price_limit = int(price_match.group(2))

    # -----------------------------
    # 2️⃣ Extract Meaningful Tokens
    # -----------------------------
    weak_words = {
        "under", "below", "above", "between",
        "for", "with", "of", "the", "and"
    }

    raw_tokens = [
        t for t in re.findall(r"\b\w+\b", query_lower)
        if t not in weak_words and not t.isdigit()
    ]

    query_tokens = set(normalize(t) for t in raw_tokens)

    # -----------------------------
    # 3️⃣ Hard Filter Products
    # -----------------------------
    candidate_indices = []

    for idx in range(len(products_df)):
        product = products_df.iloc[idx]
        name = str(product["name"]).lower()

        # ---- Price Filter ----
        if price_limit is not None:
            price = product.get("actual_price", None)
            try:
                clean_price = int(str(price).replace("₹", "").replace(",", ""))
                if clean_price > price_limit:
                    continue
            except:
                continue

        # ---- Normalize Product Tokens ----
        name_tokens = set(
            normalize(w)
            for w in re.findall(r"\b\w+\b", name)
        )

        # ---- Strict AND Match ----
        if query_tokens:
            if not query_tokens.issubset(name_tokens):
                continue

        candidate_indices.append(idx)

    # -----------------------------
    # 4️⃣ Semantic Ranking
    # -----------------------------
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    if len(candidate_indices) == 0:
        # fallback to pure semantic
        similarities = cosine_similarity(
            query_embedding,
            product_embeddings
        )[0]
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
    else:
        similarities = cosine_similarity(
            query_embedding,
            product_embeddings[candidate_indices]
        )[0]

        ranked = sorted(
            zip(similarities, candidate_indices),
            key=lambda x: x[0],
            reverse=True
        )

        ranked_indices = [idx for _, idx in ranked[:top_k]]

    # -----------------------------
    # 5️⃣ Build Final Results
    # -----------------------------
    results = []
    seen = set()

    for idx in ranked_indices:
        product = products_df.iloc[idx]
        name = str(product["name"])

        if name not in seen:
            seen.add(name)
            results.append({
                "name": name,
                "category": str(product["category_source"]),
                "price": str(product.get("actual_price", ""))
            })

    return results