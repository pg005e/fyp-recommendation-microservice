FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Artifacts are mounted at /app/artifacts in compose; override ARTIFACTS_DIR otherwise.
ENV ARTIFACTS_DIR=/app/artifacts
ENV PORT=5001
EXPOSE 5001

# gunicorn with a single worker keeps the ~15KB model loaded once in memory.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "60", "app:app"]
