# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from answer_engine import answer_question

app = FastAPI()

# Allow CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class Query(BaseModel):
    question: str

@app.post("/api/")
async def ask_virtual_ta(query: Query):
    # Call your RAG engine
    result = answer_question(query.question)

    return {
        "answer": result["answer"],
        "links": result["links"]
    }
