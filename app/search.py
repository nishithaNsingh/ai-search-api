from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_csv("data/cleaned_products.csv")
embeddings = np.load("embeddings/product_embeddings_v2.npy")

def search_products(query):
    
    query_embedding = model.encode([query])

    similarities = np.dot(embeddings, query_embedding.T).flatten()

    top_indices = similarities.argsort()[-10:][::-1]

    results = []

    for i in top_indices:
        results.append({
            "name": df.iloc[i]["name"],
            "category": df.iloc[i]["category"],
            "price": df.iloc[i]["price"]
        })

    return {"query": query, "results": results}