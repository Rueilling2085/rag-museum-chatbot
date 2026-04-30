# app.py
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import mimetypes

# Fix for Windows registry issues serving .js as text/plain
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openpyxl import Workbook

from museum_rag_core import (
    rag_answer,
    IMAGES_DIR,
    IMAGES_WEB_ROOT,       # museum_rag_core 裡已設成 "/static"
    retrieve_two_domains,
    list_artifacts_from_docs,
    find_best_artifact_match,
)

# ----------------------------------------------------------------------
# FastAPI 初始化
# ----------------------------------------------------------------------
app = FastAPI(title="Museum RAG Backend")

# CORS：允許前端（例如 localhost:5173）存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 之後要鎖特定網域再改
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# 靜態圖片掛載：把 images 資料夾掛在 IMAGES_WEB_ROOT（目前是 "/static"）
# 這樣 /static/xxx.png 就會對應到 museum-backend/images/xxx.png
# ----------------------------------------------------------------------
if os.path.exists(IMAGES_DIR):
    app.mount(
        IMAGES_WEB_ROOT,                # e.g. "/static"
        StaticFiles(directory=IMAGES_DIR),
        name="static",
    )

# ----------------------------------------------------------------------
# 前端整合：Serving Frontend from /dist
# ----------------------------------------------------------------------
# 在 Zeabur 上，前端 dist 資料夾在 backend 的上一層（同個 repo）
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "../dist")
FRONTEND_ASSETS = os.path.join(FRONTEND_DIST, "assets")

if os.path.exists(FRONTEND_ASSETS):
    # 掛載 assets -> /assets
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")

@app.get("/")
async def serve_frontend():
    """Serve the React app index.html at root."""
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Backend running, but frontend not built."}

# ----------------------------------------------------------------------
# 對話紀錄：JSONL + Excel 匯出
# ----------------------------------------------------------------------
BACKEND_ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(BACKEND_ROOT, "logs")
LOG_JSONL = os.path.join(LOG_DIR, "conversation_logs.jsonl")
LOG_XLSX = os.path.join(LOG_DIR, "conversation_logs.xlsx")


def log_event(record: Dict[str, Any]) -> None:
    """將每次對話 / 檢索記錄成一行 JSON（JSONL 格式）"""
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        record_with_time = {
            "timestamp": datetime.utcnow().isoformat(),
            **record,
        }
        with open(LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_with_time, ensure_ascii=False) + "\n")
    except Exception as e:
        # 不讓 logging 影響主流程，只印在後端 console
        print(f"[LOG] 寫入紀錄失敗：{e}")


# ----------------------------------------------------------------------
# Pydantic 請求資料模型
# ----------------------------------------------------------------------
class SearchRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    artifact_name: Optional[str] = None


# ----------------------------------------------------------------------
# 1) 先查詢「可能的文物清單」
#    前端先打這個，拿到候選文物名稱列表後，產生按鈕讓使用者選
# ----------------------------------------------------------------------
@app.post("/artifacts/search")
async def artifacts_search(payload: SearchRequest) -> Dict[str, Any]:
    """
    給一個問題，先做 RAG 檢索，回傳候選文物清單讓前端顯示按鈕。
    不直接產生回答與圖片。

    回傳格式：
    {
        "question": "...",
        "artifacts": [
            {"name": "文物A", "score": 1.0},
            {"name": "文物B", "score": 0.66},
            ...
        ]
    }
    """
    try:
        hits_basic, hits_hong, _ = retrieve_two_domains(payload.question)
        
        # 獲取模糊匹配優先權
        fuzzy_matches = find_best_artifact_match(payload.question)
        top_matched_names = [m[0] for m in fuzzy_matches if m[1] > 0.75]
        
        all_docs = (hits_basic or []) + (hits_hong or [])
        artifacts = list_artifacts_from_docs(all_docs, top_priority_names=top_matched_names)

        response = {
            "question": payload.question,
            "artifacts": artifacts,
        }

        # 記錄檢索結果
        log_event(
            {
                "endpoint": "/artifacts/search",
                "question": payload.question,
                "artifacts": artifacts,
            }
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 2) 使用者選好文物後，再拿最終答案（文字 + 可能的圖片 URL）
# ----------------------------------------------------------------------
@app.post("/chat")
async def chat(payload: ChatRequest) -> Dict[str, Any]:
    """
    前端在使用者選定文物後呼叫：
    - question: 使用者原問題
    - artifact_name: 使用者從候選清單中選擇的文物名稱（或 None -> 通用回答）

    回傳：
    {
      "answer": "...",
      "artifact_name": "實際使用的文物名稱 或 None",
      "image_url": "/static/xxx.png 或 None",   # 由 museum_rag_core.rag_answer 決定
      "artifacts": [...]                        # 同 /artifacts/search，用不到也沒關係
    }
    """
    try:
        result = rag_answer(
            question=payload.question,
            artifact_name=payload.artifact_name,
        )

        # 記錄問答結果
        log_event(
            {
                "endpoint": "/chat",
                "question": payload.question,
                "artifact_name": payload.artifact_name,
                "answer": result.get("answer"),
                "image_url": result.get("image_url"),
                "artifacts": result.get("artifacts", []),
            }
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 3) 匯出對話紀錄為 Excel
# ----------------------------------------------------------------------
@app.get("/logs/export_excel")
async def export_logs_excel():
    """
    將 logs/conversation_logs.jsonl 轉成 Excel 檔，並提供下載。
    """
    if not os.path.exists(LOG_JSONL):
        raise HTTPException(status_code=404, detail="尚未有任何對話紀錄。")

    rows = []
    with open(LOG_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        raise HTTPException(status_code=404, detail="紀錄檔為空。")

    # 建立 Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "logs"

    # 想看的欄位
    headers = [
        "timestamp",
        "endpoint",
        "question",
        "artifact_name",
        "answer",
        "image_url",
        "artifacts_json",
    ]
    ws.append(headers)

    for r in rows:
        ws.append(
            [
                r.get("timestamp", ""),
                r.get("endpoint", ""),
                r.get("question", ""),
                r.get("artifact_name", ""),
                r.get("answer", ""),
                r.get("image_url", ""),
                json.dumps(r.get("artifacts", []), ensure_ascii=False),
            ]
        )

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    wb.save(LOG_XLSX)

    return FileResponse(
        LOG_XLSX,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="conversation_logs.xlsx",
    )


# ----------------------------------------------------------------------
# 健康檢查 endpoint
# ----------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# SPA fallback：所有未匹配的路由都回傳 index.html（支援 React Router）
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Backend running, frontend not found."}
