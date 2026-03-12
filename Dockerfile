FROM registry.redhat.io/ubi9/python-311@sha256:92c7193547dcc65cff32060488f36438281ac04e50a0131f29617e85c88c4da3

WORKDIR /app

USER root
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

COPY pyproject.toml __init__.py ./
COPY components/ components/
COPY pipelines/ pipelines/

RUN chown -R 1001:1001 /app
USER 1001

RUN uv sync --no-cache --extra test

CMD ["python"]