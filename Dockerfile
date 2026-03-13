FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LLM_PROVIDER=mock
ENV SQLI_APP_PORT=5001
ENV SQLI_DB_PATH=:memory:
ENV RESULTS_DIR=/app/results
ENV PYTHONUNBUFFERED=1

EXPOSE 5001

RUN mkdir -p /app/results

CMD ["python", "scripts/demo.py"]
