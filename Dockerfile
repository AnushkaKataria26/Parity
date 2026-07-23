FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Ollama must run as a sibling service or host process, not inside this container.
# This container only encapsulates the Python dependencies and the Parity codebase.

COPY . .

ENTRYPOINT ["python", "-m", "parity.cli.main"]
