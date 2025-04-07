FROM python:3.12-slim

WORKDIR /app

# Install dependencies including CMake for building kiwipiepy and Node.js for npx
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js and npm installation
RUN node --version && npm --version && npx --version

# Install pip-tools for better dependency management
RUN pip install --no-cache-dir pip-tools

COPY requirements.txt .

# Make sure numpy is installed before other packages
RUN pip install --no-cache-dir numpy && \
    # Then install kiwipiepy and kiwipiepy-model separately first
    pip install --no-cache-dir kiwipiepy==0.20.4 kiwipiepy-model==0.20.0 && \
    # Then install the remaining packages
    grep -v "kiwipiepy\|kiwipiepy-model\|numpy" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHON_ENV=dev
ENV PYTHONUNBUFFERED=1

# 건강 체크 확인용 헬스체크 파일 추가
# RUN echo "OK" > /app/health.txt

COPY mcp_config.json /app/mcp_config.json

EXPOSE 8000

CMD ["streamlit", "run", "app_KOR.py", "--server.address", "0.0.0.0", "--server.port", "8000"] 
