# ==============================================================
# 階段一：建置 React 前端
# ==============================================================
FROM node:20-slim AS frontend-build

WORKDIR /app

# 複製前端設定檔
COPY package*.json ./
COPY vite.config.js ./
COPY index.html ./
COPY src/ ./src/
COPY public/ ./public/

# 安裝並建置
RUN npm install
RUN npm run build
# 建置結果在 /app/dist/

# ==============================================================
# 階段二：Python FastAPI 後端 + 服務前端靜態檔案
# ==============================================================
FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（rembg/Pillow 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 複製後端程式碼
COPY backend/ ./backend/

# 複製前端建置結果（放在 /app/dist/，讓 app.py 的 ../dist 路徑正確對應）
COPY --from=frontend-build /app/dist ./dist/

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r backend/requirements.txt

# 建立必要資料夾
RUN mkdir -p backend/logs backend/images/generated

EXPOSE 8080

# 啟動 FastAPI（它會同時 serve 前端靜態檔）
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
