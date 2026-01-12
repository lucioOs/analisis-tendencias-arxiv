from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import certifi
import feedparser


PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def _parse_struct_time(st) -> Optional[datetime]:
    if not st:
        return None
    try:
        return datetime(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec, tzinfo=timezone.utc)
    except Exception:
        return None


def pick_entry_datetime(entry: Any) -> datetime:
    dt = _parse_struct_time(getattr(entry, "published_parsed", None))
    if dt is None:
        dt = _parse_struct_time(getattr(entry, "updated_parsed", None))
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt


def normalize_text(s: str) -> str:
    s = str(s or "").replace("\n", " ").replace("\r", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


@dataclass
class FetchResult:
    df: pd.DataFrame
    last_dt: Optional[datetime]


def fetch_arxiv_page(query: str, start: int, max_results: int, timeout: int = 30) -> FetchResult:
    base = "https://export.arxiv.org/api/query"
    # arXiv API usa query con espacios y OR; requests lo encodea en params
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "TrendsDashboard/1.0 (contact: local)",
    }

    r = requests.get(base, params=params, headers=headers, timeout=timeout, verify=certifi.where())
    r.raise_for_status()

    feed = feedparser.parse(r.text)

    rows = []
    last_dt = None

    for e in feed.entries:
        dt = pick_entry_datetime(e)  # fecha real
        last_dt = dt if last_dt is None else min(last_dt, dt)

        title = normalize_text(getattr(e, "title", ""))
        summary = normalize_text(getattr(e, "summary", ""))
        text = (title + ". " + summary).strip()

        rows.append(
            {
                "date": dt.isoformat(),
                "text": text,
                "source": "arxiv",
                "id": getattr(e, "id", None),
                "title": title,
                "url": getattr(e, "link", None),
            }
        )

    df = pd.DataFrame(rows)
    return FetchResult(df=df, last_dt=last_dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help='Ej: "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"')
    ap.add_argument("--window_days", type=int, default=60, help="Ventana hacia atrás (días) para cubrir varios periodos")
    ap.add_argument("--page_size", type=int, default=100, help="Tamaño por página")
    ap.add_argument("--max_pages", type=int, default=20, help="Límite de páginas para evitar loops")
    ap.add_argument("--out", default=str(PROCESSED / "live_arxiv.parquet"))
    args = ap.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.window_days)

    all_parts = []
    start = 0
    pages = 0

    while pages < args.max_pages:
        res = fetch_arxiv_page(args.query, start=start, max_results=args.page_size)
        dfp = res.df

        if dfp.empty:
            break

        # parse datetime
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce", utc=True).dt.tz_convert(None)
        dfp = dfp.dropna(subset=["date", "text"])
        dfp = dfp[dfp["text"].str.len() >= 20]

        all_parts.append(dfp)

        # condición de corte: si el item más viejo de esta página ya es más viejo que cutoff, paramos
        page_min = dfp["date"].min()
        if page_min is not pd.NaT and page_min.replace(tzinfo=timezone.utc) < cutoff:
            break

        start += args.page_size
        pages += 1

    if not all_parts:
        print("[WARN] arXiv: sin datos")
        return

    df = pd.concat(all_parts, ignore_index=True)

    # dedup robusto
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    elif "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")
    else:
        df = df.drop_duplicates(subset=["text"], keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"[INFO] Exportado: {out_path} | rows={len(df)} | min={df['date'].min()} | max={df['date'].max()} | unique_days={df['date'].dt.date.nunique()}")


if __name__ == "__main__":
    main()
