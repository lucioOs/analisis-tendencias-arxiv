# src/ingest_rss.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


def _clean_html(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    # elimina tags muy básico sin depender de bs4
    out = []
    inside = False
    for ch in s:
        if ch == "<":
            inside = True
            continue
        if ch == ">":
            inside = False
            continue
        if not inside:
            out.append(ch)
    s = "".join(out)
    return " ".join(s.split()).strip()


def _pick_date(entry: dict) -> pd.Timestamp | None:
    # arXiv RSS normalmente trae published o updated
    for k in ("published", "updated"):
        v = entry.get(k)
        if v:
            ts = pd.to_datetime(v, errors="coerce", utc=True)
            if pd.notna(ts):
                return ts
    # a veces trae published_parsed/updated_parsed (struct_time)
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                dt = datetime(*v[:6], tzinfo=timezone.utc)
                return pd.Timestamp(dt)
            except Exception:
                pass
    return None


def fetch_arxiv_rss(urls: Iterable[str], days_back: int = 30) -> pd.DataFrame:
    """
    Descarga RSS de arXiv y regresa DF con:
    date (UTC), title, abstract, text, categories, link, id
    """
    try:
        import feedparser  # type: ignore
    except Exception as e:
        raise RuntimeError("Falta dependencia: pip install feedparser") from e

    now = pd.Timestamp.utcnow().tz_localize("UTC")
    cut = now - pd.Timedelta(days=int(days_back))

    rows: list[dict] = []
    for url in urls:
        feed = feedparser.parse(url)

        for entry in feed.get("entries", []):
            dt = _pick_date(entry)
            if dt is None:
                continue
            if dt < cut:
                continue

            title = _clean_html(entry.get("title", "") or "")
            # arXiv RSS suele traer summary/description con el abstract
            abstract = _clean_html(entry.get("summary", "") or entry.get("description", "") or "")
            link = (entry.get("link", "") or "").strip()

            # categorías: tags -> term
            tags = entry.get("tags", []) or []
            cats = []
            for t in tags:
                term = (t.get("term") or "").strip()
                if term:
                    cats.append(term)
            categories = ",".join(sorted(set(cats)))

            # id estable
            _id = (entry.get("id", "") or link or f"{title}|{dt.isoformat()}").strip()

            # text para TF-IDF: título + abstract (mucho mejor que texto genérico)
            text = f"{title}. {abstract}".strip()

            rows.append(
                {
                    "date": dt,
                    "title": title,
                    "abstract": abstract,
                    "text": text,
                    "categories": categories,
                    "link": link,
                    "id": _id,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df
