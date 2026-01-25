#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_dataset_focus.py

Genera un dataset histórico "enfocado" (dataset_focus.parquet) a partir del histórico completo
(dataset.parquet). El enfoque está alineado con el modo LIVE (RSS) usando:

- Filtrado por categorías tipo arXiv (cs.AI, cs.LG, stat.ML, cs.CL, cs.CV, etc.)
- Scoring por coincidencias de keywords (LLM, transformers, diffusion, RAG, etc.)
- Ventana temporal opcional
- Muestreo opcional (para no cargar millones si no hace falta)

Entrada esperada:
- Parquet con columnas típicas: ['date', 'text', 'source', 'id', 'categories']
  (Si existen title/abstract/summary, se combinan como fallback)

Salida:
- data/processed/dataset_focus.parquet
- data/processed/dataset_focus_meta.json

Uso:
  python src/build_dataset_focus.py
  python src/build_dataset_focus.py --input data/processed/dataset.parquet --min-score 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("build_dataset_focus")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# Taxonomía / Tópicos "LIVE-like"
# -----------------------------
@dataclass(frozen=True)
class TopicSpec:
    name: str
    keywords: Tuple[str, ...]


def default_topic_specs() -> List[TopicSpec]:
    """
    Tópicos alineados a lo típico del LIVE en computación/IA.
    Ajusta libremente para tu expo, pero esta base es sólida.
    """
    return [
        TopicSpec(
            "LLM_Transformers",
            (
                "llm", "large language model", "transformer", "attention",
                "bert", "gpt", "instruction tuning", "fine-tuning", "finetuning",
                "prompt", "prompting", "alignment", "rlhf", "dpo", "sft",
                "tokenizer", "context window",
            ),
        ),
        TopicSpec(
            "RAG_Retrieval",
            (
                "rag", "retrieval augmented generation", "retrieval-augmented",
                "vector database", "embedding", "embeddings", "faiss", "hnsw",
                "semantic search", "dense retrieval", "bm25", "reranker", "reranking",
                "knowledge base", "grounding",
            ),
        ),
        TopicSpec(
            "NLP",
            (
                "nlp", "natural language processing", "text classification",
                "named entity recognition", "ner", "sentiment", "summarization",
                "machine translation", "mt", "question answering",
            ),
        ),
        TopicSpec(
            "Computer_Vision",
            (
                "computer vision", "image", "vision transformer", "vit",
                "object detection", "segmentation", "yolo", "cnn", "convolution",
                "self-supervised", "contrastive", "clip",
            ),
        ),
        TopicSpec(
            "Diffusion_Generative",
            (
                "diffusion", "stable diffusion", "denoising",
                "generative", "text-to-image", "image generation", "gan",
                "variational autoencoder", "vae",
            ),
        ),
        TopicSpec(
            "Graph_ML",
            (
                "graph neural network", "gnn", "graph convolution", "gcn",
                "graph attention", "gat", "knowledge graph",
            ),
        ),
        TopicSpec(
            "Reinforcement_Learning",
            (
                "reinforcement learning", "rl", "policy gradient", "ppo",
                "q-learning", "actor-critic", "bandit",
            ),
        ),
        TopicSpec(
            "Robotics",
            (
                "robot", "robotics", "slam", "motion planning", "control",
                "manipulation", "autonomous", "navigation",
            ),
        ),
        TopicSpec(
            "Cybersecurity",
            (
                "cybersecurity", "security", "malware", "phishing",
                "intrusion", "vulnerability", "exploit", "zero-day",
                "cryptography", "encryption",
            ),
        ),
        TopicSpec(
            "Systems_Distributed",
            (
                "distributed", "distributed systems", "microservices",
                "kubernetes", "docker", "cloud", "serverless",
                "fault tolerance", "consensus", "raft", "paxos",
                "scalability", "latency",
            ),
        ),
    ]


def default_category_prefixes() -> Tuple[str, ...]:
    """
    Categorías arXiv típicas para computación/IA.
    LIVE RSS suele venir con estas (o muy cercanas).
    """
    return (
        "cs.AI",
        "cs.LG",
        "stat.ML",
        "cs.CL",
        "cs.CV",
        "cs.RO",
        "cs.IR",
        "cs.NE",
        "cs.CR",
        "cs.DC",
        "cs.SE",
        "cs.SY",
        "eess.SP",  # signal processing (a veces relevante)
    )


# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date_like(x) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x, utc=False, errors="coerce")
    except Exception:
        return None


def normalize_categories(cat) -> str:
    """
    Devuelve una cadena "normalizada" para búsquedas tipo 'cs.AI' dentro de categories.
    Soporta: list, string, NaN.
    """
    if cat is None or (isinstance(cat, float) and pd.isna(cat)):
        return ""
    if isinstance(cat, (list, tuple, set)):
        return " ".join([str(c) for c in cat if c is not None])
    return str(cat)


def pick_text_column(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    - Si existe 'text', usarla.
    - Si no, construir a partir de title/abstract/summary si están.
    """
    if "text" in df.columns:
        return df["text"].fillna("").astype(str), "text"

    fallback_cols = [c for c in ("title", "abstract", "summary") if c in df.columns]
    if not fallback_cols:
        raise ValueError(
            f"No encuentro columnas de texto. Columnas disponibles: {list(df.columns)[:60]}..."
        )

    combined = df[fallback_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return combined, "+".join(fallback_cols)


def compile_topic_regexes(topic_specs: Iterable[TopicSpec]) -> Dict[str, re.Pattern]:
    """
    Regex robusto: palabras completas cuando aplica, y tolerante a guiones/espacios.
    """
    compiled: Dict[str, re.Pattern] = {}
    for spec in topic_specs:
        escaped = []
        for kw in spec.keywords:
            kw = kw.strip().lower()
            if not kw:
                continue
            # Convertimos espacios múltiples a patrón flexible
            kw_pat = re.escape(kw).replace(r"\ ", r"[\s\-_/]+")
            escaped.append(kw_pat)
        # Unión OR
        pat = r"(" + r"|".join(escaped) + r")"
        compiled[spec.name] = re.compile(pat, flags=re.IGNORECASE)
    return compiled


def score_row(text: str, cats_norm: str, cat_prefixes: Tuple[str, ...], topic_rx: Dict[str, re.Pattern]) -> Tuple[int, List[str]]:
    """
    Score = matches_por_tópico + bonus por categorías.
    - 1 punto por cada tópico que aparezca al menos una vez en el texto.
    - +1 si cae en categorías de interés.
    """
    text = (text or "").strip()
    cats_norm = (cats_norm or "").strip()

    matched_topics: List[str] = []
    score = 0

    for topic, rx in topic_rx.items():
        if rx.search(text):
            matched_topics.append(topic)
            score += 1

    # Bonus por categorías "computación/IA"
    if cats_norm:
        if any(prefix in cats_norm for prefix in cat_prefixes):
            score += 1

    return score, matched_topics


# -----------------------------
# Main pipeline
# -----------------------------
def build_focus(
    input_path: str,
    output_path: str,
    meta_path: str,
    cat_prefixes: Tuple[str, ...],
    min_score: int,
    since: Optional[str],
    until: Optional[str],
    max_rows: Optional[int],
    sample_frac: Optional[float],
    seed: int,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el input: {input_path}")

    LOG.info("Leyendo parquet: %s", input_path)
    df = pd.read_parquet(input_path)

    if df.empty:
        raise ValueError("El dataset de entrada está vacío.")

    # Normalizar fecha
    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date_like)
    else:
        LOG.warning("No existe columna 'date'. El filtrado temporal no aplicará.")
        df["date"] = pd.NaT

    # Filtrado temporal
    if since:
        since_ts = pd.to_datetime(since, errors="raise")
        df = df[df["date"].notna() & (df["date"] >= since_ts)]
        LOG.info("Filtro since=%s -> rows=%d", since, len(df))

    if until:
        until_ts = pd.to_datetime(until, errors="raise")
        df = df[df["date"].notna() & (df["date"] <= until_ts)]
        LOG.info("Filtro until=%s -> rows=%d", until, len(df))

    # Muestreo opcional (para iterar rápido)
    if sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("--sample-frac debe estar en (0, 1].")
        df = df.sample(frac=sample_frac, random_state=seed)
        LOG.info("Sample frac=%.4f -> rows=%d", sample_frac, len(df))

    # Límite duro opcional
    if max_rows is not None:
        df = df.head(max_rows)
        LOG.info("Max rows=%d -> rows=%d", max_rows, len(df))

    # Texto principal
    text_series, text_source = pick_text_column(df)

    # Categorías normalizadas
    cats_norm = df["categories"].apply(normalize_categories) if "categories" in df.columns else pd.Series([""] * len(df))
    df["_categories_norm"] = cats_norm

    # Compilar tópicos
    topic_specs = default_topic_specs()
    topic_rx = compile_topic_regexes(topic_specs)

    # Scoring
    LOG.info("Scoring por tópicos + categorías (min_score=%d)...", min_score)

    scores: List[int] = []
    topics_list: List[List[str]] = []

    # Loop explícito (más controlable y debug-friendly)
    for t, c in zip(text_series.tolist(), df["_categories_norm"].tolist()):
        s, mt = score_row(t, c, cat_prefixes, topic_rx)
        scores.append(s)
        topics_list.append(mt)

    df["_text"] = text_series.astype(str)
    df["focus_score"] = scores
    df["focus_topics"] = topics_list

    # Filtrar por score mínimo
    before = len(df)
    df_focus = df[df["focus_score"] >= min_score].copy()
    after = len(df_focus)
    LOG.info("Filtro focus_score>=%d: %d -> %d", min_score, before, after)

    # Limpieza / columnas finales
    keep_cols = []
    for col in ["id", "date", "source", "categories", "_categories_norm", "focus_score", "focus_topics", "_text", "text"]:
        if col in df_focus.columns:
            keep_cols.append(col)

    # Garantiza que quede 'text' como campo principal si el dashboard lo espera
    if "text" not in df_focus.columns:
        df_focus["text"] = df_focus["_text"]

    # Ordenar por fecha si existe
    if "date" in df_focus.columns:
        df_focus = df_focus.sort_values("date", ascending=True)

    # Dedupe simple (por id si existe; si no, por (date,text) aproximado)
    if "id" in df_focus.columns:
        df_focus = df_focus.drop_duplicates(subset=["id"], keep="last")
    else:
        df_focus["_text_hash"] = df_focus["text"].astype(str).str.slice(0, 512)
        df_focus = df_focus.drop_duplicates(subset=["date", "_text_hash"], keep="last")
        df_focus = df_focus.drop(columns=["_text_hash"], errors="ignore")

    safe_mkdir(os.path.dirname(output_path))
    LOG.info("Guardando focus parquet: %s", output_path)
    df_focus.to_parquet(output_path, index=False)

    # Meta
    def _minmax(series: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        if series.isna().all():
            return None, None
        return str(series.min()), str(series.max())

    dmin, dmax = _minmax(df_focus["date"]) if "date" in df_focus.columns else (None, None)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "output_path": output_path,
        "rows_input_after_filters": int(before),
        "rows_output_focus": int(len(df_focus)),
        "min_score": int(min_score),
        "text_source": text_source,
        "category_prefixes": list(cat_prefixes),
        "since": since,
        "until": until,
        "sample_frac": sample_frac,
        "max_rows": max_rows,
        "date_min": dmin,
        "date_max": dmax,
        "topic_names": [t.name for t in topic_specs],
    }

    LOG.info("Guardando meta: %s", meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOG.info(
        "[OK] Focus dataset listo: rows=%d | date_min=%s | date_max=%s | text_source=%s",
        len(df_focus),
        dmin,
        dmax,
        text_source,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera dataset histórico enfocado (LIVE-like).")
    p.add_argument(
        "--input",
        default=os.path.join("data", "processed", "dataset.parquet"),
        help="Ruta al parquet histórico completo.",
    )
    p.add_argument(
        "--output",
        default=os.path.join("data", "processed", "dataset_focus.parquet"),
        help="Ruta de salida del parquet enfocado.",
    )
    p.add_argument(
        "--meta",
        default=os.path.join("data", "processed", "dataset_focus_meta.json"),
        help="Ruta de salida del meta JSON.",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="Score mínimo para conservar un registro (>=). Recomendado: 2 o 3.",
    )
    p.add_argument(
        "--since",
        default=None,
        help="Fecha mínima (YYYY-MM-DD) para filtrar histórico.",
    )
    p.add_argument(
        "--until",
        default=None,
        help="Fecha máxima (YYYY-MM-DD) para filtrar histórico.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Límite duro de filas después de filtros (útil para pruebas rápidas).",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Muestreo aleatorio (0,1]. Ej: 0.10 = 10%% para pruebas.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para muestreo.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging: DEBUG, INFO, WARNING, ERROR.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cat_prefixes = default_category_prefixes()

    build_focus(
        input_path=args.input,
        output_path=args.output,
        meta_path=args.meta,
        cat_prefixes=cat_prefixes,
        min_score=args.min_score,
        since=args.since,
        until=args.until,
        max_rows=args.max_rows,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
