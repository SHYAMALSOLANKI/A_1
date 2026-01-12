FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for some python packages if needed (e.g. gcc for compilation)
# language-tool-python requires java
RUN apt-get update && apt-get install -y \
    default-jre \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLI model during build to cache it
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-distilroberta-base')"

COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "services.stack_server.api:app", "--host", "0.0.0.0", "--port", "8000"]
