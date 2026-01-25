# src/live_runner.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

# IMPORTANTE: Idealmente, 'src.ingest_rss' debe incluir pausas internas entre URLs (rate limit).
from src.ingest_rss import fetch_arxiv_rss

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# =============================================================================
# Paths
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LIVE_DIR = DATA_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DATASET = LIVE_DIR / "live_dataset.parquet"
LIVE_META = LIVE_DIR / "live_meta.json"


# =============================================================================
# Sources
# =============================================================================
ARXIV_RSS_URLS: list[str] = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CV",
    "https://export.arxiv.org/rss/cs.SE",
    "https://export.arxiv.org/rss/cs.CR",
    "https://export.arxiv.org/rss/cs.DS",
    "https://export.arxiv.org/rss/stat.ML",
]

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _rss_url_to_cat(url: str) -> Optional[str]:
    parts = url.rstrip("/").split("/")
    if not parts:
        return None
    cat = parts[-1].strip()
    return cat or None


# =============================================================================
# Schema
# =============================================================================
REQUIRED_COLS: tuple[str, ...] = (
    "date",
    "title",
    "abstract",
    "text",
    "categories",
    "link",
    "id",
)

TEXT_COLS: tuple[str, ...] = ("title", "abstract", "text", "categories", "link", "id")


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class RunConfig:
    days_back: int = 30
    max_keep: int = 20_000
    min_rows_warn: int = 20
    write_meta: bool = True
    strict: bool = False
    log_level: str = "INFO"

    # Fallback API (arXiv API oficial)
    api_fallback: bool = True
    api_page_size: int = 200      # resultados por request/página
    api_max_total: int = 2000     # tope total acumulado para cubrir days_back
    api_timeout_sec: int = 30
    api_retries: int = 3
    api_backoff_sec: float = 3.0

    # Rate limiting (arXiv: pausas entre requests)
    api_polite_sleep_sec: float = 3.1

    # Demo mode (si todo queda vacío, rellena con “últimos N”)
    force_non_empty: bool = True
    force_min_rows: int = 50


# =============================================================================
# Logging
# =============================================================================
def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Helpers (I/O)
# =============================================================================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logging.warning("No se pudo leer parquet '%s': %s", path, e)
        return pd.DataFrame()


def _write_meta(meta_path: Path, meta: dict) -> None:
    _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))


# =============================================================================
# Helpers (DataFrame hygiene)
# =============================================================================
def _ensure_columns(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    out = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing and strict:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    for c in missing:
        out[c] = pd.NaT if c == "date" else ""

    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    for c in TEXT_COLS:
        out[c] = out[c].astype(str).fillna("")

    return out


def _normalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in TEXT_COLS:
        out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Rellenar text si viene vacío/corto
    need_text = out["text"].str.len() < 20
    if need_text.any():
        out.loc[need_text, "text"] = (
            out.loc[need_text, "title"].fillna("")
            + ". "
            + out.loc[need_text, "abstract"].fillna("")
        ).str.strip()

    # Rellenar id si viene vacío
    need_id = out["id"].str.len() == 0
    if need_id.any():
        out.loc[need_id, "id"] = out.loc[need_id, "link"]

    still_need = out["id"].str.len() == 0
    if still_need.any():
        out.loc[still_need, "id"] = (
            out.loc[still_need, "title"]
            + "|"
            + out.loc[still_need, "date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["id"], keep="last")
    out = out.drop_duplicates(subset=["link"], keep="last")
    out = out.drop_duplicates(subset=["title", "date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _trim(df: pd.DataFrame, max_keep: int) -> pd.DataFrame:
    if max_keep > 0 and len(df) > max_keep:
        return df.tail(max_keep).reset_index(drop=True)
    return df


def _filter_days_back(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    cut = now - pd.Timedelta(days=int(days_back))
    return df[df["date"] >= cut].reset_index(drop=True)


# =============================================================================
# arXiv API Logic
# =============================================================================
def _api_http_get(url: str, params: dict, timeout_sec: int) -> str:
    headers = {
        "User-Agent": "PredicTrends/1.0 (mailto:osva7777@gmail.com)",
        "Accept": "application/atom+xml,application/xml,text/xml,*/*",
    }

    if requests is None:
        import urllib.parse
        import urllib.request

        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{q}", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            return r.read().decode("utf-8", errors="replace")

    r = requests.get(url, params=params, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.text


def _fetch_arxiv_api(
    cats: Sequence[str],
    days_back: int,
    page_size: int,
    max_total: int,
    timeout_sec: int,
    retries: int,
    backoff_sec: float,
    polite_sleep_sec: float,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> pd.DataFrame:
    """
    arXiv API oficial (Atom). Paginación por `start` hasta:
    - cubrir el cutoff `days_back`, o
    - alcanzar `max_total`, o
    - recibir 0 entradas.
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        logging.error("Falta 'feedparser'. Instala: pip install feedparser")
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    cats = [c for c in cats if c]
    if not cats:
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    query = " OR ".join([f"cat:{c}" for c in cats])

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days_back))
    pagepage_size = int(max(1, page_size))
    max_total = int(max(0, max_total))

    all_rows: list[dict] = []
    start = 0

    while True:
        if max_total > 0 and len(all_rows) >= max_total:
            break

        max_results = page_size
        if max_total > 0:
            remaining = max_total - len(all_rows)
            max_results = min(max_results, remaining)

        params = {
            "search_query": query,
            "start": int(start),
            "max_results": int(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        entries: Optional[list] = None
        last_err: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                if attempt > 1:
                    sleep_s = backoff_sec * (2 ** (attempt - 2))
                    logging.warning(
                        "API retry %s/%s (sleep %.1fs)...",
                        attempt,
                        retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

                logging.info(
                    "Consultando arXiv API start=%s max_results=%s (intento %s)...",
                    start,
                    max_results,
                    attempt,
                )
                xml = _api_http_get(ARXIV_API_URL, params=params, timeout_sec=timeout_sec)

                feed = feedparser.parse(xml)
                if getattr(feed, "bozo", False):
                    logging.debug("Feedparser warning: %s", feed.get("bozo_exception"))

                entries = feed.get("entries", []) or []
                break
            except Exception as e:
                last_err = e
                entries = None

        if entries is None:
            logging.error("Error llamada API en start=%s: %s", start, last_err)
            break

        if not entries:
            break

        oldest_in_page: Optional[pd.Timestamp] = None
        page_rows: list[dict] = []

        for entry in entries:
            dt = None
            if entry.get("published"):
                dt = pd.to_datetime(entry["published"], errors="coerce", utc=True)
            elif entry.get("updated"):
                dt = pd.to_datetime(entry["updated"], errors="coerce", utc=True)

            if dt is None or pd.isna(dt):
                continue

            dt_ts = pd.Timestamp(dt)
            if oldest_in_page is None or dt_ts < oldest_in_page:
                oldest_in_page = dt_ts

            title = (entry.get("title") or "").replace("\n", " ").strip()
            abstract = (entry.get("summary") or "").replace("\n", " ").strip()

            link = entry.get("link") or ""
            for lk in entry.get("links", []) or []:
                if lk.get("rel") == "alternate" and lk.get("type") == "text/html":
                    link = lk.get("href") or link
                    break

            tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
            categories = ",".join(sorted(set(tags)))

            raw_id = entry.get("id") or link
            clean_id = raw_id.split("/")[-1] if raw_id else f"{title[:20]}|{dt_ts.isoformat()}"

            page_rows.append(
                {
                    "date": dt_ts,
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title}. {abstract}",
                    "categories": categories,
                    "link": link,
                    "id": clean_id,
                }
            )

        if not page_rows:
            break

        all_rows.extend(page_rows)

        time.sleep(float(polite_sleep_sec))

        if oldest_in_page is not None and oldest_in_page <= cutoff:
            break

        start += int(max_results)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# Main Logic
# =============================================================================
def run(cfg: RunConfig, urls: Optional[Sequence[str]] = None) -> int:
    urls = list(urls or ARXIV_RSS_URLS)

    if cfg.days_back <= 0:
        raise ValueError("days_back debe ser > 0")
    if cfg.max_keep < 0:
        raise ValueError("max_keep debe ser >= 0")
    if cfg.api_page_size <= 0:
        raise ValueError("api_page_size debe ser > 0")
    if cfg.api_max_total < 0:
        raise ValueError("api_max_total debe ser >= 0")

    logging.info("Iniciando ingesta. Fuente primaria: RSS (%s feeds)", len(urls))
    rss_df_raw = fetch_arxiv_rss(urls, days_back=int(cfg.days_back))
    rss_df = _normalize_text_fields(
        _ensure_columns(rss_df_raw if rss_df_raw is not None else pd.DataFrame(), strict=cfg.strict)
    )
    rss_df = _filter_days_back(rss_df, cfg.days_back)

    new_df = rss_df
    new_source = "rss"

    if cfg.api_fallback and rss_df.empty:
        cats = [c for c in (_rss_url_to_cat(u) for u in urls) if c]
        logging.warning("RSS vacío. Fallback API cats: %s", cats)

        api_df_raw = _fetch_arxiv_api(
            cats=cats,
            days_back=cfg.days_back,
            page_size=cfg.api_page_size,
            max_total=cfg.api_max_total,
            timeout_sec=cfg.api_timeout_sec,
            retries=cfg.api_retries,
            backoff_sec=cfg.api_backoff_sec,
            polite_sleep_sec=cfg.api_polite_sleep_sec,
        )
        api_df = _normalize_text_fields(_ensure_columns(api_df_raw, strict=False))
        api_df = _filter_days_back(api_df, cfg.days_back)

        if not api_df.empty:
            new_df = api_df
            new_source = "api"

    if cfg.force_non_empty and new_df.empty:
        cats = [c for c in (_rss_url_to_cat(u) for u in urls) if c]
        logging.warning("Dataset vacío. Forzando carga API (últimos %s registros)...", cfg.force_min_rows)

        force_df_raw = _fetch_arxiv_api(
            cats=cats,
            days_back=3650,
            page_size=min(cfg.api_page_size, max(cfg.force_min_rows, 10)),
            max_total=max(cfg.force_min_rows, 10),
            timeout_sec=cfg.api_timeout_sec,
            retries=cfg.api_retries,
            backoff_sec=cfg.api_backoff_sec,
            polite_sleep_sec=cfg.api_polite_sleep_sec,
        )
        force_df = _normalize_text_fields(_ensure_columns(force_df_raw, strict=False))
        if not force_df.empty:
            new_df = force_df.tail(int(cfg.force_min_rows)).reset_index(drop=True)
            new_source = "api_force"

    old_df_raw = _safe_read_parquet(LIVE_DATASET)
    old_df = (
        _normalize_text_fields(_ensure_columns(old_df_raw, strict=False))
        if not old_df_raw.empty
        else pd.DataFrame(columns=list(REQUIRED_COLS))
    )

    old_ids = set(old_df["id"].astype(str)) if not old_df.empty else set()
    new_ids = set(new_df["id"].astype(str)) if not new_df.empty else set()
    real_new_count = len(new_ids - old_ids)

    frames: list[pd.DataFrame] = []
    for df_ in (old_df, new_df):
        if df_ is not None and not df_.empty:
            frames.append(df_.dropna(axis=1, how="all"))

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(REQUIRED_COLS))
    merged = _dedup(merged)
    merged = _trim(merged, cfg.max_keep)

    _atomic_write_parquet(merged, LIVE_DATASET)

    if cfg.write_meta:
        meta = {
            "updated_at_utc": _now_utc_iso(),
            "days_back": int(cfg.days_back),
            "source_used": new_source,
            "total_rows": int(len(merged)),
            "new_fetched": int(len(new_df)),
            "new_unique": int(real_new_count),
            "min_date": str(merged["date"].min()) if not merged.empty else None,
            "max_date": str(merged["date"].max()) if not merged.empty else None,
            "parquet": str(LIVE_DATASET),
            "api_page_size": int(cfg.api_page_size),
            "api_max_total": int(cfg.api_max_total),
            "api_polite_sleep_sec": float(cfg.api_polite_sleep_sec),
            "api_timeout_sec": int(cfg.api_timeout_sec),
        }
        _write_meta(LIVE_META, meta)

    if real_new_count < cfg.min_rows_warn:
        logging.warning(
            "Pocos registros nuevos (únicos): %s (min_rows_warn=%s). source=%s",
            real_new_count,
            cfg.min_rows_warn,
            new_source,
        )

    logging.info("Fin. Fuente=%s | nuevos_unicos=%s | total=%s", new_source, real_new_count, len(merged))
    return 0


# =============================================================================
# CLI
# =============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingesta LIVE arXiv RSS/API -> data/live/live_dataset.parquet")

    p.add_argument("--days-back", type=int, default=30, help="Ventana de días (default: 30)")
    p.add_argument("--max-keep", type=int, default=20_000, help="Máximo de filas a conservar (0=sin recorte)")
    p.add_argument("--min-rows-warn", type=int, default=20, help="Warn si entran pocos nuevos únicos")
    p.add_argument("--no-meta", action="store_true", help="No escribir live_meta.json")
    p.add_argument("--strict", action="store_true", help="Fallar si faltan columnas requeridas")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")

    p.add_argument("--no-api", action="store_true", help="Deshabilitar fallback arXiv API")
    p.add_argument("--api-page-size", type=int, default=200, help="Resultados por página del API (default: 200)")
    p.add_argument("--api-max-total", type=int, default=2000, help="Tope total acumulado (default: 2000)")

    # Demo mode: por default está activo; aquí solo permites apagarlo
    p.add_argument("--no-force-non-empty", action="store_true", help="Deshabilitar modo demo (permitir vacío)")
    p.add_argument("--force-min-rows", type=int, default=50, help="Mínimo de filas en modo demo (default: 50)")

    p.add_argument(
        "--url",
        action="append",
        default=None,
        help="Agrega una URL RSS (se puede repetir). Si no se usa, toma el set por defecto.",
    )
    return p


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = RunConfig(
        days_back=args.days_back,
        max_keep=args.max_keep,
        min_rows_warn=args.min_rows_warn,
        write_meta=not args.no_meta,
        strict=args.strict,
        log_level=args.log_level,
        api_fallback=not args.no_api,
        api_page_size=args.api_page_size,
        api_max_total=args.api_max_total,
        force_non_empty=not bool(args.no_force_non_empty),
        force_min_rows=int(args.force_min_rows),
    )

    _setup_logging(cfg.log_level)
    urls = args.url if args.url else ARXIV_RSS_URLS
    return run(cfg, urls=urls)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
