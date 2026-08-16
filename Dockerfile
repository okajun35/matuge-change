FROM python:3.11-slim

# MediaPipe / OpenCV が要求する共有ライブラリと、H.264 再エンコード用の ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe Face Landmarker モデル（リポジトリにはコミットしないためイメージに焼く）
RUN curl -sL -o models/face_landmarker.task --create-dirs \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# pymatting の numba 関数はimport時にJITコンパイルされ、そこで一時的に ~490MB 使う。
# glibc はそのヒープをOSへ返さないので、512MB ホストでは起動直後の常駐が ~575MB まで
# 膨らみ、解析リクエストでOOM kill される（RSS 575MB → 273MB）。ここでコンパイルして
# キャッシュ（site-packages 内の .nbi/.nbc）をイメージに焼き、起動時は読むだけにする。
RUN python -c "import pymatting"

COPY backend ./backend
COPY frontend ./frontend

RUN mkdir -p data

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
