#!/bin/bash

set -e

# Nur bauen, wenn das Binary fehlt
if [ ! -f /app/llama.cpp/build/bin/llama-server ]; then
  echo "Building llama.cpp with CUDA support..."
  cd /app/llama.cpp
  mkdir -p build && cd build
  cmake .. -DLLAMA_CUDA=on -DLLAMA_BUILD_SERVER=on -DLLAMA_BUILD_TESTS=off
  cmake --build . --config Release
else
  echo "llama-server bereits vorhanden – Build übersprungen."
fi

# Server starten
# In Build-Verzeichnis wechseln und starten
cd /app/llama.cpp/build
echo "Starte llama-server..."
exec ./bin/llama-server -m /models/llm_modell.gguf -t 28 --host 0.0.0.0 --port 5001

