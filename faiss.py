import os
import json
import weaviate
from sentence_transformers import SentenceTransformer

# ------------------- CONFIG -------------------
SCRAPED_DIR = "scraped_data"
WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ConfluencePage"
TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # same used for embeddings
# ----------------------------------------------

# Initialize embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to Weaviate
client = weaviate.Client(WEAVIATE_URL)

# ------------------- SCHEMA SETUP -------------------
if not client.schema.exists(CLASS_NAME):
    schema = {
        "classes": [
            {
                "class": CLASS_NAME,
                "description": "Confluence documents with text and image references",
                "vectorizer": "none",  # we'll provide our own embeddings
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "url", "dataType": ["string"]},
                    {"name": "image_path", "dataType": ["string"]},
                ],
            }
        ]
    }
    client.schema.create(schema)

# ------------------- DATA INGESTION -------------------
def ingest_embeddings(scraped_dir):
    for root, _, files in os.walk(scraped_dir):
        for file in files:
            if file.endswith("embeddings.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                    for doc in docs:
                        client.data_object.create(
                            {
                                "text": doc["text"],
                                "url": doc["url"],
                                "image_path": doc["image_path"],
                            },
                            CLASS_NAME,
                            vector=doc["embedding"]
                        )
    print("✅ Data ingestion complete.")

# ------------------- QUERY + RAG -------------------
def query_chatbot(user_query, top_k=TOP_K):
    # Search top-k relevant docs
    results = client.query.get(CLASS_NAME, ["text", "url", "image_path"])\
        .with_near_text({"concepts": [user_query]})\
        .with_limit(top_k).do()

    docs = results["data"]["Get"][CLASS_NAME]
    context_text = "\n\n".join([d["text"] for d in docs])
    images = [d["image_path"] for d in docs]

    # Generate intelligent answer using LLM
    # Example using OpenAI GPT-4o-mini
    try:
        from openai import OpenAI
        llm_client = OpenAI()
        prompt = f"Answer the question using only the context below:\n\nContext:\n{context_text}\n\nQuestion: {user_query}"
        answer = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer_text = answer.choices[0].message["content"]
    except ImportError:
        # fallback: local LLM or just return retrieved text
        answer_text = context_text

    return {
        "answer": answer_text,
        "images": images
    }

# ------------------- MAIN -------------------
if __name__ == "__main__":
    # Step 1: Ingest data (run once)
    ingest_embeddings(SCRAPED_DIR)

    # Step 2: Query chatbot
    user_question = "How do I reset my password?"
    response = query_chatbot(user_question, top_k=3)

    print("\n🧠 Answer:\n", response["answer"])
    print("\n🖼️ Related Images:")
    for img in response["images"]:
        print(img)
