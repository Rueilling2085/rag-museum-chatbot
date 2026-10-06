# rag-museum-chatbot（博物館 AI 導覽系統）

為國立故宮博物院展覽設計並建構的 AI-native 檢索增強生成（RAG）對話系統。

![對話式導覽截圖](docs/screenshot.png)

---

## Why rag-museum-chatbot?

傳統博物館的知識傳遞長期依賴靜態標籤，但標籤受限於字數，往往只能提供文物的基本資料，難以回應來自不同知識背景的觀眾。
此外，策展人會透過展品搭配，幫助觀眾想像文物過去的真實使用情境，然而一旦館藏出現缺佚，敘事脈絡便難以延續。
為此，我們致力於建構一款對話式的 AI 導覽系統。然而在測試中發現，若直接採用通用的大型語言模型（LLM），容易因缺乏特定文化知識的訓練而產生「AI 幻覺」，反而損害了展覽的權威性與可信度。

為此，本系統以 RAG（檢索增強生成）為核心，串聯公開權威資料庫進行知識檢索，確保每一則回答都有據可查。觀眾只需用自然語言提問，系統便從已建置好的知識庫中檢索並生成回應，甚至能即時生成對應的情境圖像，讓缺佚的文物也得以還原場景。

本系統透過 RAG 架構，讓觀眾從**被動接收者**轉變為**主動知識建構者**。

---

## 核心特色（Features）

- **雙域知識庫** — 基本知識（basic）與文學故事內容（hongloumeng）分域儲存，避免向量空間語意混淆。
- **混合檢索 + 三層模糊匹配** — 結合向量語意檢索與 BM25，外加異體字正規化、簡稱匹配、拼音相似度三層機制，完美應對觀眾高度口語化的提問。
- **對話式情境圖像** — 支援根據問題脈絡與展品特性，即時生成對應的歷史/文學情境圖像。
- **嚴謹的 Chunk 實驗驗證**（研究階段）— 針對切分策略、嵌入模型（FAISS + multilingual-e5-base）設計 5 組參數並以 Recall@k 量化比較，採用最佳的 `size200_100` 設定（Recall@5 = 2.0，Std = 0.894）。
- **LLM as a Judge 自動評估**（研究階段）— 針對 141 題合成問題集進行評估，Cohen's κ 達到 0.982。

> 上面兩項「研究階段」特色，是碩士論文的實驗設計與驗證結果；本 repo 部署的 demo 為實作簡化版本，技術棧請見下方「技術棧」一節。

---

## 系統架構（Architecture，實際部署版本）

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
│  │       (Numpy 向量相似度 + BM25)               │   │
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

## 專案結構（Structure）

```text
rag-museum-chatbot/
├── src/                         # React (Vite) 前端介面（位於根目錄）
│   ├── assets/                  # 文物圖片、AI 頭像等靜態素材
│   └── App.jsx                  # 主要對話邏輯與 UI
├── backend/                     # Python FastAPI 後端
│   ├── app.py                   # API 入口
│   ├── museum_rag_core.py       # RAG 核心邏輯（檢索、生成、模糊匹配）
│   ├── data/                    # 雙域知識庫（.md 格式）
│   └── images/                  # 文物原始照片
└── index.html / vite.config.js  # 前端建置設定
```

---

## 快速開始（Quick Start）

### 前置需求
- Python 3.10+ / Node.js 18+
- OpenAI API Key / Google (Gemini) API Key

### 1. 安裝與啟動後端
在 `backend/` 目錄下建立 `.env` 檔案，填入：
```env
OPENAI_API_KEY=你的 OpenAI API Key
GOOGLE_API_KEY=你的 Google API Key（用於圖像生成，選填）
```
然後執行：
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
*（後端服務將啟動於 `http://localhost:8000`，知識庫與向量索引會在服務啟動時自動載入）*

### 2. 安裝與啟動前端
在**專案根目錄**下執行：
```bash
npm install
npm run dev
```
*（前端介面將開啟於 `http://localhost:5173`）*

---

## API 說明（API Endpoints）

- **`POST /chat`**: 對話介面主要端點。接收 `query`，回傳生成之 `response`、`contexts` 與 `image_url`。
- **`POST /image-generate`**: 根據文物名稱與情境描述即時生成情境圖像。

---

## 技術棧（Tech Stack）

| 領域 | 主要技術 |
|---|---|
| 前端介面 | React 18, Vite |
| 後端框架 | Python FastAPI |
| 語言與生成模型 | OpenAI GPT-4o-mini（文字）, Google Gemini 2.5（圖像） |
| 嵌入與檢索技術 | OpenAI text-embedding-3-small（向量嵌入）, Numpy 餘弦相似度（向量檢索）, BM25（關鍵字檢索）, TF-IDF（重排序） |
| 開發與部署平台 | 前端：Vercel / 後端：Render |

研究階段曾比較 FAISS + `intfloat/multilingual-e5-base` 等替代方案，實驗方法與量化結果詳見論文與上方「核心特色」。

---

## 授權與宣告（License & Credits）

- 資料授權：知識庫資料和來源圖片版權歸 [國立故宮博物院（Open Data）](https://digitalarchive.npm.gov.tw/opendata) 所有，僅供學術研究使用。
