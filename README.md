# Multi-Turn RAG Policy Chatbot

A multi-turn Retrieval-Augmented Generation (RAG) CLI application built with LangChain, ChromaDB, and OpenAI. It ingests complex policy PDFs and provides context-grounded answers with query reformulation and hallucination guardrails.

## Features
* **Query Reformulation:** Automatically rewrites conversational follow-ups into standalone search queries using a rolling chat history window (`window_size=6`).
* **Vector Search:** Uses OpenAI `text-embedding-3-small` embeddings and Cosine Similarity search (`hnsw:space: cosine`) in ChromaDB.
* **Strict Guardrails:** System prompts restrict the model from hallucinating or answering out-of-domain questions.
* **Sample Data Included:** Comes pre-packaged with public policy PDFs in the `policies/` directory for instant testing.

## Setup & Running

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt