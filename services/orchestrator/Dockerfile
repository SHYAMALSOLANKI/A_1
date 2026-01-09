FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the necessary files for dependency installation
# We assume build context is the monorepo root
COPY services/orchestrator/pyproject.toml /app/services/orchestrator/
COPY packages/memory /app/packages/memory
COPY packages/pb2s_prompts /app/packages/pb2s_prompts
COPY packages/evals /app/packages/evals

# Configure poetry to not create a virtualenv
WORKDIR /app/services/orchestrator
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application code
COPY services/orchestrator /app/services/orchestrator

WORKDIR /app
CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
