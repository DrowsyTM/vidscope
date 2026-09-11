FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/home/appuser/.local/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      tesseract-ocr \
      tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g 10001 -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

USER 10001:10001
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser requirements-docker.txt /home/appuser/app/requirements-docker.txt
RUN pip install --no-cache-dir --user --require-hashes -r /home/appuser/app/requirements-docker.txt

COPY --chown=appuser:appuser . /home/appuser/app
RUN set -e; \
    pip wheel --no-cache-dir --no-deps -w /tmp/wheels . && \
    HASH=$(pip hash /tmp/wheels/vidscope-*.whl | sed -n 's/^ *--hash=//p') && \
    VER=$(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") && \
    pip install --no-cache-dir --user --no-deps --no-index --find-links=/tmp/wheels --require-hashes "vidscope==$VER" --hash="$HASH" && \
    rm -rf /tmp/wheels

ENTRYPOINT ["vidscope"]
CMD ["mcp"]
