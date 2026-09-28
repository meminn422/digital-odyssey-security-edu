"""把 knowledge_base/security_knowledge.json 轉成向量，存進本地 ChromaDB。

用法：
    python build_vector_db.py

之後修改或新增 security_knowledge.json 的內容後，重新執行本腳本即可（使用 upsert，
同 id 的資料會被更新，不會重複建立）。
"""
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge_base", "security_knowledge.json")
CHROMA_PATH = os.path.join(BASE_DIR, "knowledge_base", "chroma_db")
COLLECTION_NAME = "security_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_knowledge(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    items = load_knowledge(KNOWLEDGE_PATH)
    print(f"讀取到 {len(items)} 筆知識庫資料")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    questions = [item["question"] for item in items]
    embeddings = model.encode(questions, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[item["id"] for item in items],
        embeddings=embeddings,
        documents=questions,
        metadatas=[
            {
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "keywords": ", ".join(item.get("keywords", [])),
            }
            for item in items
        ],
    )

    print(f"已寫入 {collection.count()} 筆向量到 {CHROMA_PATH}")


if __name__ == "__main__":
    main()
