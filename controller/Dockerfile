# controller/Dockerfile
FROM python:3.11-slim

# Damit Build beim setzen der Zeitzone nicht hängt
ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    default-jre \
    build-essential \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /workspace

# requirements.txt kopieren
COPY requirements.txt .

# pip updaten + Pakete installieren
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["bash"]
