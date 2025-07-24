#!/bin/bash
echo "▶ Testanfrage an llama.cpp (llm-cpu) senden..."

curl -s http://llm-cpu:5000/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Nenne drei Vorteile von erneuerbaren Energien.",
    "n_predict": 50,
    "temperature": 0.7,
    "stop": ["\\n"]
  }' | jq

echo "▶ Testanfrage an llama.cpp (llm-gpu) senden..."

curl -s http://llm-gpu:5001/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Nenne drei Vorteile von erneuerbaren Energien.",
    "n_predict": 50,
    "temperature": 0.7,
    "stop": ["\\n"]
  }' | jq
