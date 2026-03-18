# 🏛️ rag-museum-chatbot (Museum AI Guide)

A Retrieval-Augmented Generation (RAG) conversational system designed for the National Palace Museum exhibits.

🌐 [Live Demo](#) · 📖 [Documentation](#) · 🕸️ [Knowledge Base](#) · 🤝 [Contribute](#)

![MIT License](https://img.shields.io/badge/License-MIT-green.svg) ![React](https://img.shields.io/badge/React-18-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)

---

## ❓ Why rag-museum-chatbot?

Museums traditionally rely on static information labels, which are limited by word count and physical space. They often fail to answer individual questions or establish deep connections between artifacts and their historical contexts.

This system uses a RAG architecture to transform visitors from **passive recipients** into **active knowledge builders**. By asking questions in natural language, visitors receive accurate, context-aware responses and even real-time generated companion imagery.

---

## ✨ Key Features

- 🧠 **Dual-Domain Knowledge Base** — Factual knowledge (basic) and literary interpretation (hongloumeng) are stored separately to avoid semantic confusion.
- 🔍 **Hybrid Retrieval + Triple-Layer Fuzzy Matching** — Combines vector semantic search with BM25, plus variant character normalization, synonym matching, and pinyin similarity to handle highly colloquial queries.
- 🖼️ **Generative Contextual Imagery** — Generates historical or literary images based on the conversation context and artifact characteristics.
- 📊 **Rigorous Chunking Experiments** — Evaluated 5 sets of parameters using Recall@k; optimized with the `size200_100` configuration (Recall@5 = 2.0, Std = 0.894).
- ⚖️ **LLM as a Judge Evaluation** — Evaluated on 141 synthetic questions, achieving a Cohen's κ of 0.982.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────┐
│                   React (Vite) Frontend              │
│              Chat Interface + Image Display          │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────┐
│               Python FastAPI Backend                 │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              RAG Pipeline                    │   │
│  │  Query → Hybrid Retrieval → Rerank → Generate   │   │
│  │          (Vector + BM25)                      │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                   │
│  ┌──────────────▼───────────────────────────────┐   │
│  │           Dual-Domain DB                     │   │
│  │  basic: Artifact Facts                       │   │
│  │  hongloumeng: Dream of the Red Chamber Context│   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
museum-rag-guide/
├── backend/                    # Python FastAPI Backend
│   ├── main.py                 # API Entry
│   ├── rag/                    # RAG Logic (Retriever, Generator, Indexer)
│   └── knowledge_base/         # Markdown Knowledge Base with YAML tags
├── frontend/                   # React (Vite) Frontend
│   └── src/                    # UI Components & Logic
└── evaluation/                 # RAG Evaluation Scripts (Ignored on GitHub)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ / Node.js 18+
- OpenAI API Key / Google Gemini API Key

### 1. Setup Backend
Create a `.env` file in the `backend/` directory with your API keys.
```bash
cd backend
pip install -r requirements.txt
python -m rag.indexer # Build index on first run
uvicorn main:app --reload --port 8000
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Endpoints

- **`POST /chat`**: Main conversation endpoint.
- **`POST /image-generate`**: Generates historical/literary imagery.

---

## 💻 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite |
| **Backend** | Python FastAPI |
| **LLM & Generative** | OpenAI GPT-4o-mini, Google Gemini 2.5 |
| **Retrieval** | E5-base, FAISS, BM25, TF-IDF |
| **Platform** | Gemini Antigravity |

---

## 📜 License & Credits

- **Source Code**: MIT
- **Data Credits**: Knowledge base and images are courtesy of the [National Palace Museum (Open Data)](https://digitalarchive.npm.gov.tw/opendata) and are for academic research only.
