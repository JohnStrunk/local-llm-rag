FROM python:3.11 as base
#FROM python:3.11-alpine as base

############################################################
## Builder image
FROM base as builder

# Not pinning package versions
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# hadolint ignore=DL3018
# RUN apk add --no-cache \
#     g++ \
#     gcc \
#     musl-dev \
#     && rm -rf /var/cache/apk/*

# Install Python dependencies into /install of the builder image
RUN pip install --no-cache-dir pipenv==2023.9.8
COPY Pipfile Pipfile.lock /
RUN pipenv requirements > /requirements.txt
RUN mkdir /install && \
    pip install --no-cache-dir --prefix=/install -r /requirements.txt



############################################################
## Loader image
FROM base as loader

# Not pinning package versions
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies from the builder image
COPY --from=builder /install /usr/local

COPY loader_file.py \
     loader_notion.py \
     /




############################################################
## Chatbot image
FROM base as chatbot

# Copy the Python dependencies from the builder image
COPY --from=builder /install /usr/local

COPY chatbot.py /
