# src/ingest_arxiv.py
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import feedparser  # type: ignore
import pandas as pd
import requests

# =============================================================================
# Configuración Global
# =============================================================================
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# User-Agent claro (recomendado por arXiv para clientes)
USER_AGENT = "PredicTrends/1.0 (mailto:tu_email@dominio.com)"

# Rate limit “polite” (arXiv suele recomendar no bombardear; 3+ s es razonable)
POLITE_SLEEP_SEC = 3.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers: Fechas y Texto
# =============================================================================
def _parse_struct_time(st: Any) -> Optional[datetime]:
    """Convierte struct_time de feedparser a datetime UTC aware."""
    if not st:
        return None
    try:
        return datetime(
            st.tm_year,
            st.tm_mon,
            st.tm_mday,
            st.tm_hour,
            st.tm_min,
            st.tm_sec,
            tzinfo=timezone.utc,
        )
    except Exception:
        return None


def get_entry_date(entry: Any) -> datetime:
    """Extrae fecha: published -> updated -> now(UTC)."""
    dt = _parse_struct_time(getattr(entry, "published_parsed", None))
    if dt is None:
        dt = _parse_struct_time(getattr(entry, "updated_parsed", None))
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt


def normalize_text(text: Any) -> str:
    """Limpia espacios, saltos de línea y nulos."""
    s = str(text or "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s.strip()


def _save_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Guarda el DataFrame de forma atómica (tmp -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
        logger.info("Guardado seguro en: %s", path)
    except Exception as e:
        logger.error("Error guardando parquet: %s", e)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


# =============================================================================
# Clase de Respuesta
# =============================================================================
@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame
    min_date_utc: Optional[datetime]  # Fecha más antigua (min) en la página


# =============================================================================
# Lógica de Extracción (Extract)
# =============================================================================
def fetch_page(
    session: requests.Session,
    query: str,
    start: int,
    max_results: int,
    retries: int,
    backoff: float,
    timeout: int,
) -> FetchResult:
    """Descarga y parsea una página del API de arXiv."""
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/atom+xml,application/xml,text/xml,*/*",
    }

    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()

            feed = feedparser.parse(resp.text)
            entries = getattr(feed, "entries", []) or []
            if not entries:
                return FetchResult(pd.DataFrame(), None)

            rows: list[dict] = []
            min_dt: Optional[datetime] = None

            for entry in entries:
                dt_utc = get_entry_date(entry)
                if min_dt is None or dt_utc < min_dt:
                    min_dt = dt_utc

                title = normalize_text(getattr(entry, "title", ""))
                summary = normalize_text(getattr(entry, "summary", ""))

                link = normalize_text(getattr(entry, "link", ""))
                raw_id = normalize_text(getattr(entry, "id", ""))

                # id típico: http://arxiv.org/abs/XXXX.XXXXXvY
                clean_id = raw_id.split("/")[-1] if raw_id else (link or f"{title[:30]}|{dt_utc.isoformat()}")

                # categorías desde tags
                tags = getattr(entry, "tags", []) or []
                cat_terms = []
                for t in tags:
                    term = normalize_text(getattr(t, "term", ""))
                    if term:
                        cat_terms.append(term)
                categories = ",".join(sorted(set(cat_terms)))

                rows.append(
                    {
                        "date": dt_utc,  # UTC aware
                        "title": title,
                        "abstract": summary,
                        "text": f"{title}. {summary}".strip(),
                        "categories": categories,
                        "link": link,
                        "id": clean_id,
                        "source": "arxiv_api",
                    }
                )

            df = pd.DataFrame(rows)
            return FetchResult(df, min_dt)

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1)) + (random.random() * 0.5)
                logger.warning(
                    "Error en página start=%s (intento %s/%s). Reintentando en %.1fs... Error: %s",
                    start,
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                logger.error("Fallo definitivo en página start=%s: %s", start, e)
                raise

    # no debería llegar aquí
    raise RuntimeError(f"Fallo fetch_page start={start}: {last_err}")


# =============================================================================
# Lógica Principal (Main Loop)
# =============================================================================
def run_ingestion(
    query: str,
    output_file: Path,
    window_days: int,
    page_size: int,
    max_pages: int,
    incremental: bool,
) -> None:
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=int(window_days))
    logger.info("Iniciando ingesta. Ventana=%s días | Cutoff=%s", window_days, cutoff_date.isoformat())

    # Carga incremental
    existing_df = pd.DataFrame()
    if incremental and output_file.exists():
        try:
            existing_df = pd.read_parquet(output_file)
            if "date" in existing_df.columns:
                existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True, errors="coerce")
            logger.info("Cargados %s registros previos.", len(existing_df))
        except Exception as e:
            logger.warning("No se pudo leer archivo previo %s; se iniciará de cero. Error: %s", output_file, e)
            existing_df = pd.DataFrame()

    session = requests.Session()
    new_frames: list[pd.DataFrame] = []
    start = 0
    pages_processed = 0
    reached_cutoff = False

    for _ in range(int(max_pages)):
        pages_processed += 1

        result = fetch_page(
            session=session,
            query=query,
            start=start,
            max_results=int(page_size),
            retries=3,
            backoff=2.0,
            timeout=30,
        )

        if result.df.empty:
            logger.info("La API no devolvió más resultados.")
            break

        # Normaliza dates del batch
        batch = result.df.copy()
        batch["date"] = pd.to_datetime(batch["date"], utc=True, errors="coerce")
        batch = batch.dropna(subset=["date"]).reset_index(drop=True)

        # Filtra por cutoff para ahorrar memoria
        batch_cut = batch[batch["date"] >= cutoff_date]
        if not batch_cut.empty:
            new_frames.append(batch_cut)

        # Si la página ya cruzó el cutoff, detener
        if result.min_date_utc is not None and result.min_date_utc < cutoff_date:
            logger.info("Alcanzada fecha de corte (%s < %s). Deteniendo.", result.min_date_utc, cutoff_date)
            reached_cutoff = True
            break

        start += int(page_size)
        time.sleep(POLITE_SLEEP_SEC)

    if not reached_cutoff and pages_processed >= int(max_pages):
        logger.warning("Se alcanzó max_pages=%s sin llegar al cutoff. Considera subir max_pages o max_total.", max_pages)

    if not new_frames and existing_df.empty:
        logger.warning("No hay datos para guardar.")
        return

    new_df = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
    logger.info("Nuevos registros obtenidos (tras cutoff): %s", len(new_df))

    # Merge
    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df.dropna(axis=1, how="all"))
    if new_df is not None and not new_df.empty:
        frames.append(new_df.dropna(axis=1, how="all"))

    full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Limpieza final
    full_df["date"] = pd.to_datetime(full_df["date"], utc=True, errors="coerce")
    full_df = full_df.dropna(subset=["date"])
    full_df = full_df[full_df["date"] >= cutoff_date]

    # Deduplicación (id -> link -> title+date)
    before = len(full_df)
    full_df = full_df.sort_values("date", ascending=True)

    if "id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["id"], keep="last")
    if "link" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["link"], keep="last")
    if {"title", "date"}.issubset(full_df.columns):
        full_df = full_df.drop_duplicates(subset=["title", "date"], keep="last")

    logger.info("Deduplicación: %s -> %s filas.", before, len(full_df))

    # Guardado: UTC naive (documentado)
    full_df["date"] = full_df["date"].dt.tz_convert(None)

    _save_atomic_parquet(full_df.reset_index(drop=True), output_file)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta profesional de arXiv (API oficial)")
    parser.add_argument("--query", type=str, required=True, help="Query arXiv API (ej: 'cat:cs.AI')")
    parser.add_argument("--window", type=int, default=30, help="Días de historia a mantener (default: 30)")
    parser.add_argument("--output", type=str, default="data/processed/live_arxiv.parquet", help="Ruta parquet salida")
    parser.add_argument("--full-refresh", action="store_true", help="Ignora archivo existente (no incremental)")
    parser.add_argument("--page-size", type=int, default=100, help="Resultados por página (default: 100)")
    parser.add_argument("--max-pages", type=int, default=300, help="Límite de páginas (default: 300)")

    args = parser.parse_args()

    run_ingestion(
        query=args.query,
        output_file=Path(args.output),
        window_days=int(args.window),
        page_size=int(args.page_size),
        max_pages=int(args.max_pages),
        incremental=not bool(args.full_refresh),
    )


if __name__ == "__main__":
    main()
