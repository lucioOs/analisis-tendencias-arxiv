# src/ingest_arxiv.py
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import certifi
import feedparser
import pandas as pd
import requests

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers de fecha / texto
# -----------------------------
def _parse_struct_time(st) -> Optional[datetime]:
    if not st:
        return None
    try:
        return datetime(
            st.tm_year, st.tm_mon, st.tm_mday,
            st.tm_hour, st.tm_min, st.tm_sec,
            tzinfo=timezone.utc,
        )
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
    s = " ".join(s.split())
    return s.strip()

def _to_naive_utc(dt_aware: datetime) -> datetime:
    # dt_aware está en UTC; lo guardamos naive para consistencia con el resto del proyecto
    return dt_aware.astimezone(timezone.utc).replace(tzinfo=None)

def _safe_str(x: Any) -> str:
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


# -----------------------------
# Fetch
# -----------------------------
@dataclass
class FetchResult:
    df: pd.DataFrame
    page_min_dt_utc: Optional[datetime]  # el más viejo de la página (aware UTC)

def fetch_arxiv_page(
    session: requests.Session,
    query: str,
    start: int,
    max_results: int,
    timeout: int = 30,
    retries: int = 3,
    backoff_sec: float = 1.5,
    debug: bool = False,
) -> FetchResult:
    """
    Descarga una página del API de arXiv y devuelve dataframe + fecha mínima de la página.
    """
    base = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "TrendsDashboard/1.0 (contact: local)"}

    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(
                base,
                params=params,
                headers=headers,
                timeout=timeout,
                verify=certifi.where(),
            )
            r.raise_for_status()

            feed = feedparser.parse(r.text)

            # feedparser puede marcar "bozo" cuando hay algo raro en XML, pero a veces igual hay data útil.
            if getattr(feed, "bozo", 0) and debug:
                exc = getattr(feed, "bozo_exception", None)
                print(f"[DEBUG] feed.bozo=1 start={start} exc={exc}")

            entries = getattr(feed, "entries", None) or []
            if not entries:
                return FetchResult(df=pd.DataFrame(), page_min_dt_utc=None)

            rows = []
            dts: list[datetime] = []

            for e in entries:
                dt = pick_entry_datetime(e)  # aware UTC
                dts.append(dt)

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
            page_min_dt = min(dts) if dts else None
            return FetchResult(df=df, page_min_dt_utc=page_min_dt)

        except Exception as e:
            last_err = e
            if attempt < retries:
                # backoff + jitter (evita choques si arXiv rate-limitea)
                jitter = random.random() * 0.25
                time.sleep(backoff_sec * attempt + jitter)
            else:
                raise RuntimeError(f"Error consultando arXiv (start={start}): {e}") from e

    raise RuntimeError(f"Error consultando arXiv: {last_err}")


# -----------------------------
# Dedup / filtro / merge
# -----------------------------
def dedup_robusto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "id" in df.columns and df["id"].notna().any():
        df = df.drop_duplicates(subset=["id"], keep="last")
    elif "url" in df.columns and df["url"].notna().any():
        df = df.drop_duplicates(subset=["url"], keep="last")
    else:
        df = df.drop_duplicates(subset=["text"], keep="last")

    return df

def parse_and_filter(df: pd.DataFrame, cutoff_utc_aware: datetime) -> pd.DataFrame:
    """
    - parse date
    - limpia texto
    - filtra por cutoff (window_days)
    - guarda date naive
    """
    if df.empty:
        return df

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "text"])

    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df[df["text"].str.len() >= 20]

    # a naive
    df["date"] = df["date"].dt.tz_convert(None)

    cutoff_naive = _to_naive_utc(cutoff_utc_aware)
    df = df[df["date"] >= cutoff_naive]

    return df

def _normalize_prev(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el parquet previo para que merge no reviente.
    """
    if prev is None or prev.empty:
        return pd.DataFrame()

    prev = prev.copy()

    if "date" in prev.columns:
        prev["date"] = pd.to_datetime(prev["date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        prev["date"] = pd.NaT

    if "text" in prev.columns:
        prev["text"] = prev["text"].astype(str).map(normalize_text)
    else:
        prev["text"] = ""

    if "source" not in prev.columns:
        prev["source"] = "arxiv"

    for col in ["id", "title", "url"]:
        if col not in prev.columns:
            prev[col] = None

    prev = prev.dropna(subset=["date"])
    prev = prev[prev["text"].str.len() >= 20]
    return prev[["date", "text", "source", "id", "title", "url"]]


def _stats(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    if df.empty:
        return None, None, 0
    try:
        mn = df["date"].min()
        mx = df["date"].max()
        unique_days = df["date"].dt.date.nunique()
        return mn, mx, int(unique_days)
    except Exception:
        return None, None, 0


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Ingesta arXiv con auto-cutoff (paginado hasta window_days).")
    ap.add_argument("--query", required=True, help='Ej: "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"')
    ap.add_argument("--window_days", type=int, default=180, help="Ventana hacia atrás (días)")
    ap.add_argument("--page_size", type=int, default=100, help="Tamaño por página")
    ap.add_argument("--max_pages", type=int, default=200, help="Límite de seguridad de páginas (evita loops)")
    ap.add_argument("--timeout", type=int, default=30, help="Timeout HTTP (seg)")
    ap.add_argument("--retries", type=int, default=3, help="Reintentos por página")
    ap.add_argument("--backoff", type=float, default=1.5, help="Backoff base (seg)")
    ap.add_argument("--polite_sleep", type=float, default=0.25, help="Pausa entre páginas (seg) para ser amable con arXiv")
    ap.add_argument("--out", default=str(PROCESSED / "live_arxiv.parquet"), help="Salida parquet")
    ap.add_argument("--incremental", action="store_true", help="Si existe el parquet, lo mezcla (no pierdes historia).")
    ap.add_argument("--debug", action="store_true", help="Logs extra para diagnóstico.")
    args = ap.parse_args()

    window_days = max(1, int(args.window_days))
    page_size = max(1, int(args.page_size))
    max_pages = max(1, int(args.max_pages))

    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=window_days)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar previo si incremental
    prev = pd.DataFrame()
    if args.incremental and out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            prev = _normalize_prev(prev)
        except Exception:
            prev = pd.DataFrame()

    session = requests.Session()

    all_parts = []
    start = 0
    pages_used = 0
    reached_cutoff = False

    t0 = time.time()

    for _ in range(max_pages):
        pages_used += 1

        res = fetch_arxiv_page(
            session=session,
            query=args.query,
            start=start,
            max_results=page_size,
            timeout=int(args.timeout),
            retries=int(args.retries),
            backoff_sec=float(args.backoff),
            debug=bool(args.debug),
        )

        dfp = res.df
        if dfp.empty:
            # no hay más resultados
            break

        dfp2 = parse_and_filter(dfp, cutoff_utc_aware=cutoff_utc)
        if not dfp2.empty:
            all_parts.append(dfp2)

        # condición de parada: esta página ya tiene elementos más viejos que cutoff
        if res.page_min_dt_utc is not None and res.page_min_dt_utc < cutoff_utc:
            reached_cutoff = True
            break

        start += page_size
        time.sleep(max(0.0, float(args.polite_sleep)))

    df_new = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

    if df_new.empty and prev.empty:
        print("[WARN] arXiv: sin datos (nuevo y previo vacío).")
        return

    # Merge incremental
    if not prev.empty:
        df = pd.concat([prev, df_new], ignore_index=True)
    else:
        df = df_new

    # Limpieza final + dedup
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date", "text"])
    df["text"] = df["text"].astype(str).map(normalize_text)

    # Recortar a ventana (evita crecimiento infinito si incremental)
    cutoff_naive = _to_naive_utc(cutoff_utc)
    df = df[df["date"] >= cutoff_naive]

    df = dedup_robusto(df)
    df = df.sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    mn, mx, unique_days = _stats(df)
    dt = time.time() - t0

    print(
        f"[INFO] Exportado: {out_path} | rows={len(df)} | "
        f"min={mn} | max={mx} | unique_days={unique_days} | "
        f"pages_used={pages_used} | reached_cutoff={reached_cutoff} | "
        f"elapsed_sec={dt:.1f}"
    )


if __name__ == "__main__":
    main()
