import json
import re
import os
from sentence_transformers import SentenceTransformer

# Configuration
INPUT_METADATA_FILE = "metadata.json"
OUTPUT_CHUNKS_FILE = "document_chunks.json"
MAX_WORDS_PER_CHUNK = 300

# Load metadata
with open(INPUT_METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

text = metadata['text_content']
images = metadata['images']

# Step 1 – Chunk the text by paragraphs and word count
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
            "images": images  # Associate same images to each chunk
        })
        current_chunk = para + "\n\n"
        word_count = len(words)

# Append last chunk
if current_chunk:
    chunks.append({
        "text": current_chunk.strip(),
        "images": images
    })

print(f"Generated {len(chunks)} text chunks.")

# Step 2 – Generate Embeddings
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [chunk['text'] for chunk in chunks]
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Attach embeddings back to chunks
for idx, chunk in enumerate(chunks):
    chunk['embedding'] = embeddings[idx].tolist()

# Step 3 – Save final chunks
final_output = {
    "url": metadata['url'],
    "title": metadata.get('title', 'N/A'),
    "scraped_at": metadata.get('scraped_at', 'N/A'),
    "chunks": chunks
}

with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as out_f:
    json.dump(final_output, out_f, indent=2)

print(f"Saved chunked data with embeddings to {OUTPUT_CHUNKS_FILE}")
