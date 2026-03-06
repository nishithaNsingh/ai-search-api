from sentence_transformers import SentenceTransformer

print("Downloading model...")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("Model downloaded successfully.")