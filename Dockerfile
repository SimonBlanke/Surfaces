FROM python:3.10-slim

COPY pyproject.toml .
COPY src/ src/

RUN python -m pip install ".[test]"
