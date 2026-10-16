FROM python:3.11-slim

# Hugging Face Spaces runs containers as an arbitrary non-root user (uid 1000).
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time. A failed prefetch must not break the
# build — the weights then download at first startup instead.
RUN python -c "import laya; laya.Router(preload=True, device='cpu')" || echo "laya prefetch failed; weights will download at first startup"

COPY main.py dino.html ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
