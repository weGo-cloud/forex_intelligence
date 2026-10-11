FROM python:3.12-slim

WORKDIR /app

# Install the project in non-editable mode using only pyproject.toml first,
# so dependency layers are cached independently of application code changes.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
