# src/ingest_rss.py
from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text: Any) -> str:
    """
    Limpieza ligera (sin bs4) para campos RSS:
    - unescape de entidades HTML
    - elimina tags
    - normaliza whitespace
    """
    if text is None:
        return ""
    s = str(text)

    # Unescape & normaliza <br>
    s = html.unescape(s)
    s = s.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")

    # Remove tags
    s = _TAG_RE.sub(" ", s)

    # Normalize whitespace
    s = _WS_RE.sub(" ", s).strip()
    return s


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    """
    Convierte diferentes formatos a pd.Timestamp UTC.
    Regresa None si no puede parsear.
    """
    if not value:
        return None

    # feedparser a veces da struct_time en *_parsed
    if isinstance(value, (tuple, list)) and len(value) >= 6:
        try:
            dt = datetime(*value[:6], tzinfo=timezone.utc)
            return pd.Timestamp(dt)
        except Exception:
            return None

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def pick_entry_date(entry: dict[str, Any]) -> pd.Timestamp | None:
    """
    arXiv RSS suele traer 'published' o 'updated' y/o sus versiones *_parsed.
    """
    for k in ("published", "updated"):
        ts = _to_utc_timestamp(entry.get(k))
        if ts is not None:
            return ts

    for k in ("published_parsed", "updated_parsed"):
        ts = _to_utc_timestamp(entry.get(k))
        if ts is not None:
            return ts

    return None


def extract_categories(entry: dict[str, Any]) -> str:
    """
    arXiv RSS: entry.tags = [{"term": "..."}]
    Devuelve string CSV ordenado y sin duplicados.
    """
    tags = entry.get("tags") or []
    cats: set[str] = set()

    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, dict):
                term = str(t.get("term") or "").strip()
                if term:
                    cats.add(term)

    return ",".join(sorted(cats))


def fetch_arxiv_rss(urls: Iterable[str], days_back: int = 30, timeout: int = 20) -> pd.DataFrame:
    """
    Descarga RSS de arXiv y regresa DF con columnas:
      date (UTC), title, abstract, text, categories, link, id

    Notas:
    - 'date' siempre queda tz-aware UTC.
    - text = title + ". " + abstract (para TF-IDF).
    - dedup por id/link al final.
    """
    if days_back <= 0:
        raise ValueError("days_back debe ser > 0")
    if timeout <= 0:
        raise ValueError("timeout debe ser > 0")

    try:
        import feedparser  # type: ignore
    except Exception as e:
        raise RuntimeError("Falta dependencia: pip install feedparser") from e

    # IMPORTANTE: evita tz_localize sobre tz-aware (tu error anterior)
    now_utc = pd.Timestamp.now(tz="UTC")
    cut = now_utc - pd.Timedelta(days=int(days_back))

    rows: list[dict[str, Any]] = []

    for url in urls:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "arxiv-trends/1.0"}, timeout=timeout)
        except TypeError:
            # Algunas versiones de feedparser no aceptan timeout/request_headers
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning("RSS error url=%s: %s", url, e)
            continue

        entries = feed.get("entries") or []
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            dt = pick_entry_date(entry)
            if dt is None or dt < cut:
                continue

            title = clean_html(entry.get("title"))
            abstract = clean_html(entry.get("summary") or entry.get("description"))
            link = str(entry.get("link") or "").strip()
            categories = extract_categories(entry)

            # ID estable (prioriza entry.id, luego link, luego fallback)
            raw_id = str(entry.get("id") or "").strip()
            stable_id = (raw_id or link or f"{title}|{dt.isoformat()}").strip()

            # Text para TF-IDF
            text = f"{title}. {abstract}".strip()

            rows.append(
                {
                    "date": dt,
                    "title": title,
                    "abstract": abstract,
                    "text": text,
                    "categories": categories,
                    "link": link,
                    "id": stable_id,
                }
            )

    df = pd.DataFrame(rows, columns=["date", "title", "abstract", "text", "categories", "link", "id"])
    if df.empty:
        return df

    # Normalización final
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Limpieza extra por seguridad
    for c in ("title", "abstract", "text", "categories", "link", "id"):
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Dedup robusta (newest wins)
    df = df.sort_values("date").drop_duplicates(subset=["id"], keep="last")
    df = df.drop_duplicates(subset=["link"], keep="last")
    df = df.drop_duplicates(subset=["title", "date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    return df
