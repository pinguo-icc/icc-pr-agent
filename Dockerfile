FROM mirror.camera360.com/base/python:3.11-slim
WORKDIR /app
ENV PIP_PROGRESS_BAR=off \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pip>=24.2"

# Copy source first so setuptools can discover packages
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY server.py pr.py pr-review.yaml ./
COPY skills/ skills/
COPY templates/ templates/
VOLUME ["/app/.pr_reviews"]
EXPOSE 8000
CMD ["python", "server.py"]
