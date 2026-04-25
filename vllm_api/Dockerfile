FROM vllm/vllm-openai:latest

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface_hub

COPY entrypoint.sh /app/entrypoint.sh
COPY adapters.sh /app/adapters.sh
RUN chmod +x /app/entrypoint.sh /app/adapters.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]