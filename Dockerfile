FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# Cloud Run: OpenAI-only by default; set DISABLE_OLLAMA=false locally if you need Ollama.
ENV DISABLE_OLLAMA=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY rag ./rag
COPY rag_local ./rag_local
COPY frontend ./frontend

EXPOSE 8080

# Cloud Run injects PORT; default 8080 for local docker run.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
