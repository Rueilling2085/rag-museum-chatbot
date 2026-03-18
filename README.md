# 🏛️ rag-museum-chatbot (博物館 AI 導覽系統)

為國立故宮博物院展覽設計並建構的 AI-native 檢索增強生成（RAG）對話系統。

🌐 [Live Demo](#) · 📖 [Documentation](#) · 🕸️ [Knowledge Base](#) · 🤝 [Contribute](#)

![MIT License](https://img.shields.io/badge/License-MIT-green.svg) ![React](https://img.shields.io/badge/React-18-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)

---

## ❓ Why rag-museum-chatbot?

歷史文化類博物館長期以「靜態資訊標籤」作為主要導覽媒介，但標籤受限於字數與實體空間，無法回應每位觀眾個別的好奇心，也難以建立文物與歷史情境的深度連結。

本系統透過 RAG 架構，讓觀眾從**被動接收者**轉變為**主動知識建構者**。只要用自然語言提問，系統便會從專屬知識庫中檢索資料生成回答，甚至能即時生成對應的情境圖像。

---

## ✨ 核心特色 (Features)

- 🧠 **雙域知識庫** — 事實性知識 (basic) 與文學詮釋 (hongloumeng) 分域儲存，避免向量空間語意混淆。
- 🔍 **混合檢索 + 三層模糊匹配** — 結合向量語意檢索與 BM25，外加異體字正規化、簡稱匹配、拼音相似度三層機制，完美應對觀眾高度口語化的提問。
- 🖼️ **對話式情境圖像** — 支援根據問題脈絡與展品特性，即時生成對應的歷史/文學情境圖像。
- 📊 **嚴謹的 Chunk 實驗驗證** — 設計 5 組切分參數並以 Recall@k 量化比較，採用最佳的 `size200_100` 設定 (Recall@5 = 2.0，Std = 0.894)。
- ⚖️ **LLM as a Judge 自動評估** — 針對 141 題合成問題集進行評估，Cohen's κ 達到 0.982。

---

## 🏗️ 系統架構 (Architecture)

```text
┌─────────────────────────────────────────────────────┐
│                   React (Vite) 前端                  │
│              對話介面 + 圖像顯示                      │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────┐
│               Python FastAPI 後端                    │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              RAG Pipeline                    │   │
│  │  查詢 → 混合檢索 → TF-IDF 重排序 → LLM 生成  │   │
│  │          (向量 + BM25)                        │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                   │
│  ┌──────────────▼───────────────────────────────┐   │
│  │           雙域知識庫                          │   │
│  │  basic 域：文物事實性資料                     │   │
│  │  hongloumeng 域：紅樓夢文學脈絡               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 📂 專案結構 (Structure)

> **⚠️ 注意事項**：目前 Github Repository 主分支對應的是 `frontend/` (前端) 的程式碼。

```text
museum-rag-guide/
├── backend/                    # Python FastAPI 後端
│   ├── main.py                 # API 入口
│   ├── rag/                    # RAG 核心邏輯 (檢索、生成、索引、模糊匹配)
│   └── knowledge_base/         # 雙域知識庫 (.md 格式，含 YAML 標記)
├── frontend/                   # React (Vite) 前端介面
│   └── src/                    # UI 元件與主要邏輯
└── evaluation/                 # RAG 評估腳本 (Chunk 實驗、大模型評分)
```

---

## 🚀 快速開始 (Quick Start)

### 前置需求
- Python 3.10+ / Node.js 18+
- OpenAI API Key / Google Gemini API Key

### 1. 安裝與設定後端
在 `backend/` 目錄下建立 `.env` 檔案並填入您的 API Keys。
```bash
cd backend
pip install -r requirements.txt
python -m rag.indexer # 首次執行需建置知識庫索引
uvicorn main:app --reload --port 8000
```
*(後端服務將啟動於 `http://localhost:8000`)*

### 2. 安裝與啟動前端
```bash
cd frontend
npm install
npm run dev
```
*(前端介面將開啟於 `http://localhost:5173`)*

---

## 🔌 API 說明 (API Endpoints)

- **`POST /chat`**: 對話介面主要端點。接收 `query`，回傳生成之 `response`、`contexts` 與 `image_url`。
- **`POST /image-generate`**: 根據文物名稱與情境描述即時生成情境圖像。

---

## 💻 技術棧 (Tech Stack)

| 領域 | 主要技術 |
|---|---|
| **前端介面** | React 18, Vite |
| **後端框架** | Python FastAPI |
| **語言與生成模型** | OpenAI GPT-4o-mini (文字), Google Gemini 2.5 (圖像) |
| **嵌入與檢索技術** | intfloat/multilingual-e5-base, FAISS (向量), BM25 (關鍵字), TF-IDF (重排序) |
| **開發與部署平台** | Gemini Antigravity |

---

## 📜 授權與宣告 (License & Credits)

- **原始碼授權**: [MIT License](LICENSE)
- **資料授權**: 知識庫資料和來源圖片版權歸 [國立故宮博物院 (Open Data)](https://digitalarchive.npm.gov.tw/opendata) 所有，僅供學術研究使用。
