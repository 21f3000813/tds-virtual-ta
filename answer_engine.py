# answer_engine.py

import faiss
import json
import openai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_FILE = "embeddings/faiss_index.index"
METADATA_FILE = "embeddings/metadata.json"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

def get_question_embedding(question: str):
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return np.array(response.data[0].embedding).astype('float32')

def retrieve_relevant_chunks(question: str, k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    q_embed = get_question_embedding(question)
    D, I = index.search(np.array([q_embed]), k)

    retrieved = [metadata[i]["text"] for i in I[0] if i < len(metadata)]
    return retrieved

def generate_answer(question: str, context_chunks):
    system_prompt = (
        "You are a helpful Virtual TA for the Tools in Data Science course (Jan 2025).\n"
        "Answer the question based on the context below. If unsure, say you don't know.\n"
        "Always provide links to helpful Discourse or course resources, if any."
    )

    context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\n{context_text}"}
    ]

    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content.strip()

def get_links_from_context(context_chunks):
    links = []
    for chunk in context_chunks:
        for line in chunk.splitlines():
            if "http" in line:
                links.append({
                    "url": line.strip(),
                    "text": line.strip()
                })
    # Return unique links only (max 3)
    seen = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen:
            unique_links.append(link)
            seen.add(link["url"])
        if len(unique_links) == 3:
            break
    return unique_links

def answer_question(question: str):
    chunks = retrieve_relevant_chunks(question)
    answer = generate_answer(question, chunks)
    links = get_links_from_context(chunks)
    return {
        "answer": answer,
        "links": links
    }

# Test this
if __name__ == "__main__":
    test_question = "Should I use gpt-3.5-turbo or gpt-4o-mini for this project?"
    result = answer_question(test_question)
    print("\nAnswer:\n", result["answer"])
    print("\nLinks:\n", result["links"])
