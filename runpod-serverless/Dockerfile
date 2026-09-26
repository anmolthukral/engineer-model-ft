# Build from the engineer-model-ft/ directory (context needs engineering-model-v4.gguf
# and Modelfile.finetuned at the repo root):
#   docker buildx build --platform linux/amd64 -f runpod-serverless/Dockerfile \
#     -t <registry>/<repo>:<tag> --push .
FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod-serverless/requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY engineering-model-v4.gguf Modelfile.finetuned ./
COPY runpod-serverless/handler.py runpod-serverless/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Bake both models into the image so a cold-started worker never needs network
# access to become ready: start the server, register/pull the models, stop it.
# (Single RUN so the backgrounded server survives across the whole step.)
RUN ollama serve & \
    OLLAMA_PID=$! && \
    until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done && \
    ollama create engineering-tone -f Modelfile.finetuned && \
    ollama pull nomic-embed-text && \
    kill "$OLLAMA_PID"

ENV RAG_DATABASE_URL=""

ENTRYPOINT ["./entrypoint.sh"]
