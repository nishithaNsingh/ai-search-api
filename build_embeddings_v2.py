import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("data/cleaned_products.csv")

# Combine important searchable text
df["combined_text"] = (
    df["name"].fillna("") + " " +
    df["category_source"].fillna("")
)

print("Loading model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Generating embeddings...")
embeddings = model.encode(
    df["combined_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save("embeddings/product_embeddings.npy", embeddings)

print("Embeddings saved.")