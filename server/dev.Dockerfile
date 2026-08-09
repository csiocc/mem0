FROM python:3.12

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first for better caching
COPY server/requirements.txt .
RUN pip install -r requirements.txt

# Install the fork's mem0 SDK into site-packages (not editable): the compose
# volume mount shadows /app, so runtime code must not live under /app.
# SDK changes under mem0/ therefore need an image rebuild (--build).
COPY pyproject.toml poetry.lock README.md /tmp/mem0-src/
COPY mem0 /tmp/mem0-src/mem0
RUN pip install /tmp/mem0-src[graph] && rm -rf /tmp/mem0-src

# Copy server code
COPY server .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
