from fastapi import FastAPI
from app.search import search_products

app = FastAPI()

@app.get("/search")
def search(query: str):
    return {
        "query": query,
        "results": search_products(query)
    }