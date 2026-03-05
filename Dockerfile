FROM python:3.11-slim-bookworm

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP server
COPY mcp_server/ ./mcp_server/
COPY .env.example ./.env.example

# Railway sets PORT env var
ENV PORT=8000

CMD ["python", "mcp_server/qwen_image_server.py"]
