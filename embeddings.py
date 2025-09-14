import json
import re
import os
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = "./scraped_data"
MAX_WORDS_PER_CHUNK = 300
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Iterate over all folders in scraped_data
for folder_name in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder_name)
    metadata_file = os.path.join(folder_path, "metadata.json")

    if not os.path.isfile(metadata_file):
        print(f"Skipping {folder_name}: metadata.json not found.")
        continue

    print(f"Processing {folder_name}")

    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    text = metadata.get('text_content', '')
    images = metadata.get('images', [])
    url = metadata.get('url', 'N/A')

    # Chunk the text
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 0]

    chunks = []
    current_chunk = ""
    word_count = 0

    for para in paragraphs:
        words = para.split()
        if word_count + len(words) <= MAX_WORDS_PER_CHUNK:
            current_chunk += para + "\n\n"
            word_count += len(words)
        else:
            chunks.append({
                "text": current_chunk.strip(),
                "images": images
            })
            current_chunk = para + "\n\n"
            word_count = len(words)

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "images": images
        })

    print(f"Generated {len(chunks)} text chunks for {folder_name}")

    # Generate embeddings
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    for idx, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[idx].tolist()

    # Save structured output
    final_output = {
        "url": url,
        "title": metadata.get('title', 'N/A'),
        "scraped_at": metadata.get('scraped_at', 'N/A'),
        "chunks": chunks
    }

    output_file = os.path.join(folder_path, "document_chunks.json")
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(final_output, out_f, indent=2)

    print(f"Saved {output_file}")
