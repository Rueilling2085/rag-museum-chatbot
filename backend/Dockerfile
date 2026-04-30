FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（rembg 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有後端檔案
COPY . .

# 建立 logs 和 images/generated 資料夾
RUN mkdir -p logs images/generated

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
