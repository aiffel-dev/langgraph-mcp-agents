FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHON_ENV=dev
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["streamlit", "run", "app_KOR.py", "--server.address", "0.0.0.0", "--server.port", "8000"] 
