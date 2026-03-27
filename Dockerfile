FROM python:3.11-slim AS builder

WORKDIR /app

ENV PIP_PROGRESS_BAR=off \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN pip install --no-cache-dir pip>=24.2

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY src/ src/
COPY server.py pr.py pr-review.yaml ./
COPY skills/ skills/
COPY templates/ templates/

VOLUME ["/app/.pr_reviews"]
EXPOSE 8000

CMD ["python", "server.py"]
