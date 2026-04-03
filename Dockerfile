ARG PYTHON_VERSION=3.13.2
FROM python:${PYTHON_VERSION}-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}"

WORKDIR /workspace

COPY requirements.lock.txt /tmp/requirements.lock.txt
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip==26.0.1 setuptools==81.0.0 wheel \
    && pip install -r /tmp/requirements.lock.txt

CMD ["bash"]
