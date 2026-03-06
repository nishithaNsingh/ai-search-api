import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading model and data...")

model = SentenceTransformer("all-mpnet-base-v2")

product_embeddings = np.load("embeddings/product_embeddings_v2.npy")

products_df = pd.read_csv("data/cleaned_products.csv")

print("Model and data loaded.")