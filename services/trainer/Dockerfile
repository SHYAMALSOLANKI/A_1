FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependencies
COPY services/trainer/pyproject.toml /app/services/trainer/
COPY packages/memory /app/packages/memory
COPY packages/evals /app/packages/evals
COPY packages/pb2s_prompts /app/packages/pb2s_prompts

# Configure poetry
WORKDIR /app/services/trainer
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy app code
COPY services/trainer /app/services/trainer

CMD ["python", "main.py"]
