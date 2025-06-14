# embedder.py

import json
import os
import faiss
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
import numpy as np

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_SOURCES = [
    "data/discourse",             # Folder with multiple .jsons
    "data/course_content.json"    # Single file for course content
]

EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.json")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000

def chunk_text(text: str) -> List[str]:
    # Break long content into smaller paragraphs of CHUNK_SIZE
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

def read_json_files(folder_path) -> List[str]:
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "posts" in data:  # Discourse threads
                for post in data["posts"]:
                    text = post.get("content", "").strip()
                    if text:
                        chunks.extend(chunk_text(text))

    return chunks

def read_json_file(path: str) -> List[str]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):  # course content
        for item in data:
            title = item.get("title", "")
            url = item.get("url", "")
            if title or url:
                content = f"{title}\n{url}".strip()
                chunks.append(content)

    return chunks

def get_embedding(text: str) -> List[float]:
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def create_faiss_index(text_chunks: List[str]):
    dimension = 1536  # for text-embedding-3-small
    index = faiss.IndexFlatL2(dimension)
    metadata = []

    print("Creating embeddings and building FAISS index...")
    for i, text in enumerate(tqdm(text_chunks)):
        embedding = get_embedding(text)
        if embedding:
            index.add(np.array([embedding]).astype('float32'))
            metadata.append({"text": text})

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved index to {INDEX_FILE}")
    print(f"✅ Saved metadata to {METADATA_FILE}")

if __name__ == "__main__":
    all_chunks = []

    for source in DATA_SOURCES:
        if os.path.isdir(source):
            all_chunks.extend(read_json_files(source))
        elif os.path.isfile(source):
            all_chunks.extend(read_json_file(source))

    print(f"Total chunks: {len(all_chunks)}")
    create_faiss_index(all_chunks)
