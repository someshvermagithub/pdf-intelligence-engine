# 📚 RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange?style=for-the-badge)
![Mistral AI](https://img.shields.io/badge/LLM-Mistral_AI-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-yellow?style=for-the-badge)

An advanced **AI-powered Retrieval-Augmented Generation (RAG) Research Assistant** built with **LangChain, Mistral AI, ChromaDB, and Streamlit**.

Upload PDFs, create vector embeddings, perform semantic search, and chat intelligently with your documents using modern LLM-powered retrieval pipelines.

---

# 🚀 Features

- ✅ Multi-PDF Upload Support
- ✅ Intelligent Text Chunking
- ✅ Vector Embeddings using Mistral AI
- ✅ Semantic Similarity Search
- ✅ MMR (Max Marginal Relevance) Retrieval
- ✅ Persistent ChromaDB Storage
- ✅ Context-Aware Question Answering
- ✅ Streamlit Chat Interface
- ✅ Adjustable Retrieval Settings
- ✅ Session-based Chat History
- ✅ Retrieved Source Inspection
- ✅ Scalable RAG Architecture

---

# 🧠 How It Works

```text
PDF Upload
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
Chroma Vector Storage
    ↓
Semantic Retrieval
    ↓
LLM Response Generation
```

---

# 🛠 Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Framework | LangChain |
| UI | Streamlit |
| Vector Database | ChromaDB |
| LLM | Mistral AI |
| Embeddings | Mistral Embeddings |
| Document Loader | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |

---

# 📂 Project Structure

```bash
RAG-Research-Assistant/
│
├── chroma_db/                 # Persistent vector database
│
├── app.py                     # Main Streamlit application
├── requirements.txt
├── README.md
├── .env
│
└── assets/
```

---

# ⚡ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/someshvermagithub/pdf-intelligence-engine.git

cd rag-research-assistant
```

---

## 2️⃣ Create Virtual Environment

### Using uv (Recommended)

```bash
uv venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

# 🌐 Live Demo

🚀 Try the application here:

[🔗 Live App](https://pdf-intelligence-engine.streamlit.app/)

---

# 📌 Deployment

This project can be deployed on:

- Streamlit Cloud

---

# 📸 Application Overview

## Upload PDFs → Build Vector Database → Chat with Documents

![RAG Workflow](https://img.shields.io/badge/RAG-Pipeline-success)

---

# 💬 Example Queries

```text
• Summarize this research paper
• What are the key findings?
• Explain chapter 3 in simple terms
• Compare the methodologies discussed
• What limitations are mentioned?
• Extract important conclusions
```

---

# ⚙️ Core Retrieval Pipeline

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k_value,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

---

# 🧩 Core Application Workflow

```python
docs = retriever.invoke(query)

context = "\n\n".join([
    doc.page_content
    for doc in docs
])

final_prompt = prompt.invoke({
    "context": context,
    "question": query
})

response = llm.invoke(final_prompt)
```

---

# 📦 Key Dependencies

```txt
langchain
langchain-core
langchain-community

langchain-mistralai
langchain_huggingface
langchain_google_genai

chromadb
sentence-transformers

pypdf
unstructured
beautifulsoup4
lxml

tiktoken
python-dotenv
pandas
numpy

fastapi
uvicorn
streamlit

tqdm
requests
```

---

# 🔥 Current Capabilities

| Feature | Status |
|---|---|
| PDF Chat | ✅ |
| Semantic Search | ✅ |
| MMR Retrieval | ✅ |
| Persistent Storage | ✅ |
| Multi-Document Support | ✅ |
| Streamlit UI | ✅ |
| Source Display | ✅ |
| Chat History | ✅ |

---

# 🚧 Planned Improvements

- Hybrid Search (BM25 + Vector)
- MultiQuery Retriever
- Re-ranking Pipeline
- Citation-aware Responses
- Streaming Responses
- Local LLM Support
- Agentic RAG Workflows
- Web Search Integration
- Research Paper Summarization
- FastAPI Backend
- Docker Deployment
- Authentication System

---

# 🧪 Future Architecture Vision

```text
User Query
    ↓
Query Expansion
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Context Compression
    ↓
LLM Reasoning
    ↓
Citation-Aware Response
```

---

# 📊 Why This Project Matters

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- LLM Application Development
- Semantic Search Systems
- AI-Powered Research Tools
- Production-ready Streamlit Applications

Perfect for:

- AI/ML Portfolios
- Resume Projects
- Research Assistants
- Enterprise Knowledge Bases
- Intelligent Document QA Systems

---

# 👨‍💻 Author

## Somesh Verma

BTech CSE (AI & ML) Student  
AI Engineer • RAG Developer • LLM Enthusiast

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Contribute improvements

---

# 📜 License

This project is licensed under the MIT License.

---



# 🧠 Built With Modern AI Stack

```text
LangChain + Mistral AI + ChromaDB + Streamlit
```
