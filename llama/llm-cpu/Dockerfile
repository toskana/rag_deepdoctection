# llm/Dockerfile – llama.cpp mit CMake (CPU-only)
FROM debian:bullseye-slim

# Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# llama.cpp klonen und mit cmake bauen
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . --config Release

# Arbeitsverzeichnis für Einstieg setzen
WORKDIR /app/llama.cpp
