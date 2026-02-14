import os
import time
import numpy as np
import faiss
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# -------------------- SETUP --------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

openai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://aipipe.org/openai/v1"
)

# -------------------- DATA --------------------
DOCS = [
    {
        "id": i,
        "content": f"Ticket {i}: Login failed due to wrong password. Reset steps provided.",
        "metadata": {"source": "support.csv"}
    }
    for i in range(84)
]

# -------------------- EMBEDDINGS --------------------
def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        vectors.extend([d.embedding for d in response.data])
    return np.array(vectors, dtype=np.float32)

print("🔹 Computing document embeddings...")
doc_embeddings = embed_texts([d["content"] for d in DOCS])
faiss.normalize_L2(doc_embeddings)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

print(f" FAISS index loaded with {index.ntotal} documents")

# -------------------- RETRIEVAL --------------------
def retrieve(query: str, k: int) -> List[Dict]:
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, k)
    results = []

    for rank, idx in enumerate(ids[0]):
        cosine = float(scores[0][rank])
        normalized = (cosine + 1) / 2  # [-1,1] -> [0,1]

        doc = DOCS[idx]
        results.append({
            "id": doc["id"],
            "score": round(normalized, 4),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    return results

# -------------------- RERANK --------------------
def rerank(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    for doc in docs:
        prompt = f"""
Query: "{query}"
Document: "{doc['content']}"

Rate the relevance of this document to the query on a scale of 0 to 10.
Respond with ONLY a number.
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )

        raw = response.choices[0].message.content.strip()
        try:
            score = min(max(float(raw) / 10, 0), 1)
        except:
            score = 0.5

        doc["score"] = round(score, 4)

    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:top_k]

# -------------------- API --------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "GET":
        return "OK", 200
    return search()

@app.route("/search", methods=["POST"])
def search():
    start = time.time()
    data = request.get_json(force=True)

    query = data.get("query", "").strip()
    k = int(data.get("k", 7))
    rerank_flag = bool(data.get("rerank", True))
    rerank_k = int(data.get("rerankK", 4))

    if not query:
        return jsonify({
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(DOCS)
            }
        })

    results = retrieve(query, k)

    reranked = False
    if rerank_flag and results:
        results = rerank(query, results, rerank_k)
        reranked = True

    latency = int((time.time() - start) * 1000)

    return jsonify({
        "results": results,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCS)
        }
    })

# -------------------- RUN (RENDER COMPATIBLE) --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
