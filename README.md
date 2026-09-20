# Multi-Document Analyzer

### Retrieval-Augmented Generation for Grounded Document Intelligence

Smart Document Analyzer is a full-stack **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents, ingest webpages, and ask natural-language questions about their knowledge base.

Instead of relying solely on an LLM's pretrained knowledge, the system retrieves relevant information from indexed sources, reranks the retrieved content, applies relevance and grounding checks, and generates answers using only the retrieved context.

The project is designed around one core principle:

> **Retrieve the right information first. Generate only from verified context.**

---

## Overview

Large Language Models are powerful, but they can produce incorrect or unsupported information when answering questions about private or domain-specific documents.

This project addresses that problem using a multi-stage RAG architecture.

Users can:

* Upload multiple PDF documents
* Ingest webpages directly from URLs
* Organize documents into isolated workspaces
* Ask questions using natural language
* Ask follow-up questions conversationally
* View retrieved source passages
* See document and page-level attribution
* Compare information across multiple sources
* Receive grounded answers
* Evaluate retrieval quality
* Benchmark the RAG pipeline

The system combines **semantic retrieval, lexical retrieval, diversity-aware selection, cross-encoder reranking, relevance filtering, and answer-grounding validation**.

---

# Key Features

## 📄 Multi-Document PDF Ingestion

Upload multiple PDF files into a workspace.

The ingestion pipeline:

1. Loads PDF pages
2. Extracts document text
3. Adds source and page metadata
4. Detects coarse document sections
5. Splits documents into overlapping chunks
6. Generates embeddings
7. Stores chunks in ChromaDB
8. Makes the workspace searchable

Default chunking configuration:

```text
Chunk size:     500 characters
Chunk overlap:   80 characters
```

Each chunk retains metadata such as:

* Source filename
* Page number
* Section
* Document metadata
* Vector similarity information

---

# 🌐 Webpage Ingestion

Documents do not have to be PDFs.

The system can ingest webpages directly from URLs.

```text
URL
 ↓
HTTP Request
 ↓
Readable Content Extraction
 ↓
HTML Cleaning
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Store
```

Web content is extracted using:

* `requests`
* `readability-lxml`
* `BeautifulSoup`

The extracted webpage content is limited to a configured maximum size before indexing.

---

# 🧠 Hybrid Retrieval

The retrieval layer does not depend on a single search strategy.

It combines:

### Dense Semantic Retrieval

Sentence Transformers generate embeddings using:

```text
all-MiniLM-L6-v2
```

These embeddings are stored in:

```text
ChromaDB
```

Semantic retrieval allows the system to find conceptually related passages even when the exact words in the query do not appear in the document.

---

### BM25 Keyword Retrieval

The system also maintains a lexical retrieval path using:

```text
rank-bm25
```

BM25 is useful when exact terms, names, technical expressions, or keyword matches are important.

The two retrieval strategies are combined before reranking.

```text
                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Semantic Search          BM25 Search
      ChromaDB               Keywords
          │                     │
          └──────────┬──────────┘
                     ▼
             Candidate Pool
```

---

# 🎯 Maximum Marginal Relevance

Retrieved vector candidates are processed using **Maximum Marginal Relevance (MMR)**.

MMR helps balance:

* Relevance to the query
* Diversity between retrieved chunks

This reduces the probability of returning many nearly identical passages while missing other useful evidence.

Conceptually:

```text
High relevance
      +
Low redundancy
      ↓
Better candidate set
```

---

# 🔄 Cross-Encoder Reranking

After hybrid retrieval, candidates are reranked using a Cross-Encoder:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates:

```text
(Query, Document Chunk)
```

pairs and produces relevance scores.

The pipeline therefore becomes:

```text
Query
  ↓
Dense Retrieval
  +
BM25 Retrieval
  ↓
Candidate Pool
  ↓
MMR Selection
  ↓
Cross-Encoder Reranking
  ↓
Deduplication
  ↓
Relevant Context
```

If the Cross-Encoder is unavailable, the system contains a lexical fallback ranking strategy.

---

# 🛡️ Grounded Generation

The generation layer is explicitly instructed to use **only the retrieved context**.

The system's RAG prompt contains grounding rules such as:

* Do not use outside knowledge
* Do not rely on model memory
* Do not introduce unsupported facts
* Ignore unrelated retrieved sections
* Synthesize rather than copy raw chunks
* Return an explicit fallback when the answer is not present

When the system cannot establish sufficient evidence, it can return:

```text
I could not find this information in the indexed sources.
```

This makes the system substantially more controlled than a simple:

```text
Retrieve → Send to LLM → Generate
```

pipeline.

---

# 🔍 Retrieval Confidence & Relevance Filtering

Retrieved chunks pass through additional checks before reaching the generation stage.

The system evaluates factors including:

* Vector similarity
* Query-term evidence
* Topic relevance
* Source dominance
* Missing query terms
* Context grounding
* Retrieval confidence

The pipeline can isolate the most relevant source when one document clearly dominates the retrieved evidence.

---

# 🧩 Query Expansion

The system performs lightweight query expansion for important technical concepts.

For example:

```text
transformer
```

can be expanded with related retrieval terms such as:

```text
attention
multi-head
transformer architecture
```

Similarly:

```text
LLM
```

can be expanded toward:

```text
large language models
language
text
```

This improves lexical and semantic retrieval for common technical terminology.

---

# 💬 Conversational Follow-Up Questions

The application supports conversational context.

For example:

```text
User:
What is self-attention?

Assistant:
Self-attention allows...

User:
What are its advantages?

Assistant:
...
```

The system can detect follow-up references and rewrite queries when additional context is necessary.

Conversation history is bounded to avoid continuously increasing the request context.

---

# 🗂️ Workspace-Based Knowledge Bases

Documents are organized into named workspaces.

Example:

```text
workspace: ai-research
```

can contain:

```text
Attention_2000.pdf
LLMs_2000.pdf
GenAI_2000.pdf
RAG_2000.pdf
```

Another workspace can contain a completely different collection.

The workspace architecture allows different knowledge bases to be indexed and queried independently.

```text
Workspace A
 ├── document_1.pdf
 ├── document_2.pdf
 └── webpage_1

Workspace B
 ├── document_3.pdf
 └── document_4.pdf
```

The active workspace is managed through a thread-local workspace manager.

---

# 📚 Source Attribution

Retrieved information is returned together with source metadata.

A source can contain:

```text
Source:
Attention_2000.pdf

Page:
2

Preview:
Attention mechanisms are widely used...
```

For webpages, the system can return:

```text
Source:
https://example.com/article

Type:
Webpage
```

This allows users to inspect where an answer originated.

---

# 🗃️ Source Management

The FastAPI backend maintains metadata about indexed sources using SQLite and SQLAlchemy.

Tracked information includes:

* Workspace
* Source type
* Source name
* Source URL
* Indexing status
* Indexing timestamp
* Ingestion history
* Ingestion errors

This creates a distinction between:

```text
RAG Vector Store
        +
Application Metadata Database
```

---

# 🧪 RAG Evaluation

The project includes a dedicated evaluation framework.

The evaluation dataset contains questions with expected:

* Source document
* Expected answer
* Key facts

Example:

```python
{
    "question": "What is self-attention?",
    "expected_source": "Attention_2000.pdf",
    "expected_answer": "...",
    "key_facts": [
        "each element attends to other elements",
        "captures long-range dependencies"
    ]
}
```

This makes it possible to evaluate the RAG system rather than relying only on subjective manual testing.

---

# 📊 Retrieval Metrics

The project evaluates retrieval using metrics such as:

### Hit@1

Whether the expected document appears first.

### Hit@3

Whether the expected document appears within the top three results.

### Hit@5

Whether the expected document appears within the top five results.

### Mean Reciprocal Rank

Measures how highly the expected source is ranked.

```text
MRR = average(1 / rank)
```

### Retrieval Latency

The evaluation framework also measures retrieval response time.

---

# 🧠 Answer Grounding Evaluation

The project also evaluates whether generated answers are supported by retrieved context.

The grounding benchmark uses sentence embeddings to estimate whether answer sentences are semantically supported by the retrieved context.

This provides an additional evaluation layer:

```text
Retrieval Quality
       +
Answer Grounding
       ↓
RAG Quality
```

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   React Frontend    │
                         │   Vite + Zustand    │
                         └──────────┬──────────┘
                                    │
                                    │ HTTP / JSON
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI API     │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          PDF Ingestion       URL Ingestion       Chat API
                 │                  │                  │
                 └────────────┬─────┴──────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Document Pipeline │
                    │                     │
                    │ Load → Clean        │
                    │ Chunk → Embed       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │   Vector Storage    │
                    └──────────┬──────────┘
                               │
                         Query │
                               ▼
                    ┌─────────────────────┐
                    │ Hybrid Retrieval    │
                    │                     │
                    │ Dense + BM25        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       MMR           │
                    │ Diversity Selection │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cross-Encoder       │
                    │    Reranking        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevance /         │
                    │ Grounding Checks    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Groq LLM         │
                    │  Answer Generation  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Sources    │
                    └─────────────────────┘
```

---

# 🔬 RAG Pipeline

The complete query pipeline can be summarized as:

```text
User Question
      │
      ▼
Query Preprocessing
      │
      ▼
Query Expansion
      │
      ├─────────────────────┐
      ▼                     ▼
Dense Retrieval          BM25 Retrieval
      │                     │
      └──────────┬──────────┘
                 ▼
          Candidate Pool
                 │
                 ▼
              MMR
                 │
                 ▼
       Cross-Encoder Reranker
                 │
                 ▼
           Deduplication
                 │
                 ▼
        Topic Relevance Filter
                 │
                 ▼
       Retrieval Context
                 │
                 ▼
       Grounding Evaluation
                 │
                 ▼
            Groq LLM
                 │
                 ▼
        Grounded Answer
                 │
                 ▼
        Source Attribution
```

---

# 🤖 Language Model

The current configuration uses Groq through LangChain:

```text
langchain-groq
```

Configured model:

```text
openai/gpt-oss-20b
```

Generation parameters include:

```text
Temperature: 0.1
Max tokens: 1024
Request timeout: 90 seconds
```

The low temperature is intentional because the application prioritizes grounded and deterministic responses over creative generation.

---

# 🔢 Embedding Model

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is used for:

* Document embeddings
* Query embeddings
* Semantic retrieval
* Embedding cache generation
* Grounding evaluation

Generated embeddings are cached locally to reduce unnecessary recomputation.

---

# 💾 Vector Database

The project uses:

```text
ChromaDB
```

with persistent storage.

Default location:

```text
data/vector_store/
```

Each workspace maps to its own Chroma collection.

---

# 🗄️ Application Database

The application layer uses:

```text
SQLite
+
SQLAlchemy
```

The database stores:

* Workspaces
* Indexed sources
* Ingestion history

Database location:

```text
backend/database/rag_app.sqlite3
```

---

# 🖥️ Frontend

The current frontend is built using:

* React 19
* Vite
* Zustand
* JavaScript
* CSS

The frontend provides:

* Workspace management
* PDF upload
* URL ingestion
* Source listing
* Conversational chat
* Retrieved source display
* Backend status handling
* Error handling
* Request retries
* Request timeout handling

The UI follows a three-panel research workspace layout:

```text
┌──────────────┬──────────────────────────┬─────────────────┐
│              │                          │                 │
│   Sources    │       Chat Area          │    Insights     │
│   Workspace  │                          │    / Sources    │
│              │                          │                 │
│              │                          │                 │
└──────────────┴──────────────────────────┴─────────────────┘
```

---

# ⚙️ Backend

The backend is built with:

```text
FastAPI
```

Responsibilities include:

* PDF ingestion
* URL ingestion
* Chat requests
* Source management
* Workspace handling
* RAG execution
* Conversation memory
* Request logging
* Error handling
* SQLite persistence

---

# 🔌 API

## Health Check

```http
GET /
```

Response:

```json
{
  "status": "online",
  "message": "RAG Analyzer API is running"
}
```

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Ingest PDF

```http
POST /ingest/pdf
```

Multipart form data:

```text
files: <PDF files>
workspace_name: ai-research
```

---

## Ingest Webpage

```http
POST /ingest/url
```

Example:

```json
{
  "url": "https://example.com/article",
  "workspace_name": "ai-research"
}
```

---

## Chat

```http
POST /chat
```

Example:

```json
{
  "question": "What is self-attention?",
  "workspace_name": "ai-research",
  "conversation_id": "conversation-001"
}
```

Response contains:

```json
{
  "success": true,
  "request_id": "api-...",
  "answer": "...",
  "sources": [],
  "summary": null,
  "workspace": "ai-research",
  "rewritten_query": null,
  "query_was_rewritten": false
}
```

---

## List Sources

```http
GET /sources?workspace_name=ai-research
```

Returns indexed sources belonging to the selected workspace.

---

# 🛠️ Tech Stack

## AI / RAG

* Python
* LangChain
* LangChain Core
* LangChain Community
* LangChain Groq
* Sentence Transformers
* Cross-Encoder
* BM25

## Retrieval

* ChromaDB
* Dense Vector Search
* BM25
* Maximum Marginal Relevance
* Cross-Encoder Reranking
* Relevance Filtering
* Query Expansion

## Document Processing

* PyPDF
* PyMuPDF
* LangChain PDF loaders
* BeautifulSoup
* readability-lxml
* Recursive Character Text Splitter

## Backend

* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy
* SQLite
* Python Multipart

## Frontend

* React 19
* Vite
* Zustand
* JavaScript
* CSS

## Infrastructure / Utilities

* python-dotenv
* Requests
* Joblib
* NumPy

---

# 📂 Project Structure

```text
Smart-document-analyzer-through-RAG/
│
├── backend/
│   ├── api/
│   │   ├── chat.py
│   │   ├── health.py
│   │   ├── ingestion.py
│   │   └── sources.py
│   │
│   ├── database/
│   │   ├── crud.py
│   │   ├── db.py
│   │   └── rag_app.sqlite3
│   │
│   ├── models/
│   │   ├── database.py
│   │   └── schemas.py
│   │
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── ingestion_service.py
│   │   └── source_service.py
│   │
│   ├── utils/
│   │   ├── error_handlers.py
│   │   ├── logging.py
│   │   └── serializers.py
│   │
│   └── server.py
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   ├── common/
│   │   │   ├── insights/
│   │   │   ├── layout/
│   │   │   └── sources/
│   │   ├── config/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── store/
│   │   └── styles/
│   │
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   ├── config.py
│   ├── document_index.py
│   ├── embeddings.py
│   ├── pdf_structure.py
│   ├── pipeline.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── utils.py
│   ├── vector_store.py
│   ├── web_loader.py
│   └── workspace_manager.py
│
├── data/
│   ├── pdf/
│   ├── text_files/
│   ├── vector_store/
│   └── embed_cache/
│
├── notebook/
│   ├── document.ipynb
│   └── pdf_loader.ipynb
│
├── evaluation.py
├── retrieval_benchmark.py
├── rag_quality_benchmark.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Install:

* Python 3.12+
* Node.js
* npm
* Git
* Groq API key

---

# 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Smart-document-analyzer-through-RAG-main
```

---

# 2. Backend Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

# 3. Configure Environment Variables

Create:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

The RAG pipeline requires a valid Groq API key for answer generation.

> Never commit `.env` or API keys to GitHub.

---

# 4. Start the Backend

From the project root:

```bash
uvicorn backend.server:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

---

# 5. Start the Frontend

Open a second terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create:

```text
.env
```

or configure the Vite environment with:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

The Vite development server will normally run at:

```text
http://localhost:5173
```

---

# 6. Use the Application

### Create / Select a Workspace

Example:

```text
ai-research
```

### Upload Documents

Upload one or more PDFs.

### Add Web Sources

Paste a webpage URL.

### Ask Questions

Examples:

```text
What is attention?
```

```text
What are the three components of attention?
```

```text
What is self-attention?
```

```text
What are the advantages of transformers?
```

```text
How are LLMs trained?
```

### Ask Follow-Up Questions

```text
What is self-attention?
```

Then:

```text
What are its advantages?
```

The system can use the previous conversation to resolve the reference.

---

# 🧪 Running Evaluation

The repository contains multiple evaluation and benchmarking scripts.

Run the main evaluation:

```bash
python evaluation.py
```

Retrieval benchmarking:

```bash
python retrieval_benchmark.py
```

RAG quality benchmarking:

```bash
python rag_quality_benchmark.py
```

These tools can evaluate:

* Retrieval accuracy
* Hit@1
* Hit@3
* Hit@5
* Mean Reciprocal Rank
* Retrieval latency
* Evidence support
* Answer grounding

---

# 📈 Example Evaluation Output

The project tracks results in a format similar to:

```text
Question: What is self-attention?

Evidence support: 73%

Raw Hit@1: 1
Final Hit@1: 1

Raw Hit@3: 1
Final Hit@3: 1

MRR: ...
Latency: ...
```

The exact scores depend on:

* Indexed documents
* Retrieval configuration
* Embedding model
* Reranker
* Query
* LLM response
* Current vector store contents

---

# 🔐 Security Considerations

The application currently uses local SQLite and local vector storage.

For production deployment, additional controls should be considered:

* Authentication
* Authorization
* Per-user workspaces
* File size limits
* URL allowlisting
* SSRF protection for arbitrary URL ingestion
* Rate limiting
* API key protection
* Persistent secret management
* Isolated vector-store access
* Secure CORS configuration

The current application should therefore be considered a **research / development system rather than a hardened production SaaS platform**.

---

# ⚠️ Current Limitations

The project is actively developed and has several limitations.

### Retrieval

* Retrieval quality depends on embedding and reranker models.
* Chunking is currently based primarily on character-based splitting.
* Complex document structures may not always be preserved perfectly.
* PDF extraction quality depends on the source PDF.

### Generation

* Answers depend on Groq API availability.
* Extremely ambiguous questions can still produce weak retrieval.
* Grounding checks are heuristic rather than a formal proof of factual correctness.

### Storage

* ChromaDB is locally persisted.
* SQLite is local to the application.
* Multi-user production storage is not implemented.

### Web Ingestion

* Webpage extraction depends on the target website's HTML structure.
* JavaScript-heavy websites may not expose all content through simple HTTP fetching.
* Arbitrary URL ingestion requires additional SSRF/security controls for production deployment.

### Scalability

* The current architecture is designed primarily for research and development.
* Cross-encoder reranking can become computationally expensive as candidate counts increase.
* Local embedding generation and vector storage would require additional infrastructure for large-scale deployments.

---

# 🗺️ Roadmap

### Retrieval

* [x] Dense semantic retrieval
* [x] BM25 retrieval
* [x] Hybrid retrieval
* [x] MMR selection
* [x] Cross-encoder reranking
* [x] Deduplication
* [x] Query expansion
* [x] Relevance filtering
* [x] Retrieval diagnostics

### RAG

* [x] Grounded generation
* [x] Source attribution
* [x] Context compression
* [x] Retrieval confidence
* [x] Answer grounding checks
* [x] Follow-up query rewriting
* [x] Multi-source retrieval

### Documents

* [x] PDF ingestion
* [x] Webpage ingestion
* [x] Page-level metadata
* [x] Workspace isolation
* [x] Source management
* [x] Ingestion history

### Evaluation

* [x] Retrieval benchmark
* [x] Hit@1
* [x] Hit@3
* [x] Hit@5
* [x] MRR
* [x] Latency measurement
* [x] Evidence-support evaluation

### Future

* [ ] Authentication
* [ ] Multi-user workspaces
* [ ] Cloud vector database
* [ ] Persistent conversation storage
* [ ] Advanced document parsing
* [ ] Table-aware retrieval
* [ ] Better PDF layout understanding
* [ ] Streaming responses
* [ ] Production observability
* [ ] Automated RAG evaluation dashboards
* [ ] Document-level access control

---

# 🎯 Design Philosophy

This project deliberately separates **retrieval**, **ranking**, **grounding**, and **generation**.

Instead of:

```text
Question
   ↓
LLM
   ↓
Answer
```

the system follows:

```text
Question
   ↓
Query Processing
   ↓
Hybrid Retrieval
   ↓
Candidate Selection
   ↓
Reranking
   ↓
Relevance Filtering
   ↓
Grounding Validation
   ↓
Context Construction
   ↓
LLM Generation
   ↓
Answer + Sources
```

This architecture makes the system easier to:

* Debug
* Benchmark
* Evaluate
* Extend
* Optimize
* Reason about

---

# 📚 Example Knowledge Base

The repository includes example AI/ML documents such as:

```text
Attention_2000.pdf
GenAI_2000.pdf
LLMs_2000.pdf
RAG_2000.pdf
```

These can be used to test questions involving:

* Attention mechanisms
* Self-attention
* Transformers
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation

---

# 🤝 Contributing

Contributions are welcome.

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Make your changes and run the relevant tests/benchmarks.

Commit:

```bash
git add .
git commit -m "feat: add your feature"
```

Push:

```bash
git push origin feature/your-feature
```

Then open a pull request.

---

# 📄 License

Add the project's chosen license here.

For example:

```text
MIT License
```

---

# 👨‍💻 Author

**Ayush Kottary**

Built as a hands-on exploration of:

```text
Retrieval-Augmented Generation
            +
Information Retrieval
            +
Natural Language Processing
            +
Vector Search
            +
LLM Applications
            +
Full-Stack Engineering
```

---

# 📌 Project Status

**Research / Development**

The project currently implements a functional end-to-end RAG system with hybrid retrieval, reranking, grounding controls, document ingestion, web ingestion, workspaces, source attribution, conversational querying, and evaluation tooling.

It is designed as both a usable document-question-answering application and an experimental platform for studying and improving RAG quality.
