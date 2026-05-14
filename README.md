# RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-yellow)



An AI-powered Retrieval-Augmented Generation (RAG) application for intelligent PDF research, summarization, semantic search, and question answering.

## Features

- PDF document ingestion
- Intelligent text chunking
- Vector embeddings using Mistral AI
- Semantic similarity search
- MMR Retrieval
- MultiQuery Retrieval
- AI-powered summarization
- Context-aware question answering
- ChromaDB vector storage

## Tech Stack

- Python
- LangChain
- ChromaDB
- Mistral AI
- Sentence Transformers
- PyMuPDF

## Current Status

🚧 Project is currently under active development.

Planned features:
- Streamlit UI
- Chat history memory
- Hybrid retrieval
- Web search integration
- Research citation support
- Multi-document querying

## Project Structure

```bash
project/
│
├── document loaders/
├── chroma_db/
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
uv pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

## Run Project

```bash
python main.py
```

## Future Improvements

- Agentic RAG
- Hybrid Search
- Re-ranking
- Streaming Responses
- Local LLM Support
- Citation-aware Answers

## Author

Somesh Verma
