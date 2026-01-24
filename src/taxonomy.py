# src/taxonomy.py
from __future__ import annotations

# Macro-áreas (keywords) — filtro PLN previo a TF-IDF
# Puedes ajustar keywords sin tocar el dashboard.
MACRO_AREAS: dict[str, list[str]] = {
    "IA / Machine Learning": [
        "machine learning", "deep learning", "neural", "transformer", "llm", "foundation model",
        "classification", "regression", "fine-tuning", "reinforcement learning", "rl",
    ],
    "NLP / LLM": [
        "natural language", "nlp", "language model", "llm", "token", "prompt", "instruction",
        "summarization", "translation", "retrieval", "rag",
    ],
    "Visión / CV": [
        "computer vision", "vision", "image", "video", "object detection", "segmentation",
        "yolo", "diffusion", "generative", "captioning",
    ],
    "Sistemas / Distribuidos": [
        "distributed", "microservices", "kubernetes", "cloud", "latency", "throughput",
        "scalability", "replication", "consensus", "fault tolerance",
    ],
    "Redes / Seguridad": [
        "network", "routing", "sdn", "firewall", "intrusion", "malware", "security",
        "cryptography", "authentication", "privacy",
    ],
    "Bases de datos / IR": [
        "database", "sql", "index", "query", "retrieval", "information retrieval",
        "vector database", "embedding", "search",
    ],
}
