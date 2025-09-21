import os
import json
import chromadb
import pickle

SCRAPED_DIR = "scraped_data"
CHROMA_DIR = "chroma_index"

def load_all_embeddings(scraped_dir):
    embeddings = []
    metadatas = []
    ids = []

    id_counter = 0

    for root, dirs, files in os.walk(scraped_dir):
        for file in files:
            if file == "embeddings.json":
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    embeddings.append(item["embedding"])
                    metadatas.append({
                        "text": item["text"],
                        "url": item["url"],
                        "image_path": item["image_path"]
                    })
                    ids.append(str(id_counter))
                    id_counter += 1

    return embeddings, metadatas, ids


def build_chroma_index(embeddings, metadatas, ids):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(name="confluence_docs")

    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ ChromaDB index built with {len(ids)} items.")
    print(f"✅ Saved at: {CHROMA_DIR}")


if __name__ == "__main__":
    embeddings, metadatas, ids = load_all_embeddings(SCRAPED_DIR)
    print(f"Loaded {len(ids)} embeddings.")
    build_chroma_index(embeddings, metadatas, ids)
