import os
import json
import faiss
import numpy as np
import pickle

# Path where scraped data folders are stored
SCRAPED_DIR = "scraped_data"
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

def load_all_embeddings(scraped_dir):
    embeddings = []
    metadata = []
    id_counter = 0

    for root, dirs, files in os.walk(scraped_dir):
        for file in files:
            if file == "embeddings.json":
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    emb = np.array(item["embedding"], dtype="float32")
                    embeddings.append(emb)

                    metadata.append({
                        "id": id_counter,
                        "text": item["text"],
                        "url": item["url"],
                        "image_path": item["image_path"]
                    })
                    id_counter += 1

    return np.vstack(embeddings), metadata


def build_faiss_index(embeddings, metadata):
    dim = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(dim)  # L2 similarity index
    index.add(embeddings)  # add all vectors

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata mapping
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")
    print(f"✅ Metadata saved to {METADATA_FILE}")
    print(f"Total vectors indexed: {len(metadata)}")


if __name__ == "__main__":
    embeddings, metadata = load_all_embeddings(SCRAPED_DIR)
    print(f"Loaded {len(metadata)} embeddings.")
    build_faiss_index(embeddings, metadata)
