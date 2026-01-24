# src/live_runner.py
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.ingest_rss import fetch_arxiv_rss


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DATASET = PROCESSED_DIR / "live_dataset.parquet"
LIVE_META = PROCESSED_DIR / "live_meta.json"


# -----------------------------
# Sources (RSS)
# -----------------------------
ARXIV_RSS_URLS: list[str] = [
    # Core (los que ya tenías)
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.CL",
    # Opcionales recomendados para que “se mueva” más en demo
    "https://arxiv.org/rss/cs.CV",
    "https://arxiv.org/rss/cs.SE",
    "https://arxiv.org/rss/cs.CR",
    "https://arxiv.org/rss/cs.DS",
    "https://arxiv.org/rss/stat.ML",
]


# -----------------------------
# Config
# -----------------------------
REQUIRED_COLS = ["date", "title", "abstract", "text", "categories", "link", "id"]


@dataclass(frozen=True)
class RunConfig:
    days_back: int = 30           # ventana de descarga en RSS
    max_keep: int = 20000         # recorte para no crecer infinito
    min_rows_warn: int = 20       # warning si viene casi vacío
    write_meta: bool = True
    strict: bool = False          # si True: falla si faltan columnas base


# -----------------------------
# Helpers
# -----------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _ensure_columns(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    out = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing and strict:
        raise ValueError(f"Faltan columnas requeridas en dataset live: {missing}")

    # crea faltantes como strings vacíos (menos date)
    for c in missing:
        if c == "date":
            out[c] = pd.NaT
        else:
            out[c] = ""

    # tipos básicos
    out["link"] = out["link"].astype(str)
    out["id"] = out["id"].astype(str)
    out["title"] = out["title"].astype(str)
    out["abstract"] = out["abstract"].astype(str)
    out["text"] = out["text"].astype(str)
    out["categories"] = out["categories"].astype(str)

    # date a datetime utc
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)

    # elimina registros sin date
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    return out


def _normalize_and_build_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que text sea útil para TF-IDF:
    text = title + ". " + abstract (si text viene vacío o pobre).
    """
    out = df.copy()

    # limpia whitespace
    for c in ["title", "abstract", "text", "categories", "link", "id"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # reconstruye text si hace falta
    text_len = out["text"].astype(str).str.len()
    need = text_len < 20
    if need.any():
        out.loc[need, "text"] = (
            out.loc[need, "title"].astype(str).fillna("")
            + ". "
            + out.loc[need, "abstract"].astype(str).fillna("")
        ).str.strip()

    # id fallback si viene vacío
    id_len = out["id"].astype(str).str.len()
    empty_id = id_len == 0
    if empty_id.any():
        out.loc[empty_id, "id"] = out.loc[empty_id, "link"].astype(str)

    # si aún queda id vacío, usa combinación estable
    id_len2 = out["id"].astype(str).str.len()
    empty_id2 = id_len2 == 0
    if empty_id2.any():
        out.loc[empty_id2, "id"] = (
            out.loc[empty_id2, "title"].astype(str)
            + "|"
            + out.loc[empty_id2, "date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype(str)
        )

    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicación robusta:
    1) id
    2) link
    3) (title, date) como último fallback
    """
    out = df.copy()

    # order: newest wins
    out = out.sort_values("date").reset_index(drop=True)

    # dedup id
    if "id" in out.columns:
        out = out.drop_duplicates(subset=["id"], keep="last")

    # dedup link
    if "link" in out.columns:
        out = out.drop_duplicates(subset=["link"], keep="last")

    # fallback
    if "title" in out.columns and "date" in out.columns:
        out = out.drop_duplicates(subset=["title", "date"], keep="last")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def _trim(df: pd.DataFrame, max_keep: int) -> pd.DataFrame:
    if max_keep and len(df) > int(max_keep):
        return df.tail(int(max_keep)).reset_index(drop=True)
    return df


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(df: pd.DataFrame, out_path: Path, meta_path: Path | None = None) -> None:
    print(f"[live_runner] saved: {out_path}")
    print(f"[live_runner] rows: {len(df)}")
    if len(df) > 0 and "date" in df.columns:
        print(f"[live_runner] min: {df['date'].min()}")
        print(f"[live_runner] max: {df['date'].max()}")
    if meta_path is not None:
        print(f"[live_runner] meta: {meta_path}")


def _fetch_new(urls: Iterable[str], days_back: int) -> pd.DataFrame:
    df = fetch_arxiv_rss(urls, days_back=int(days_back))
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)
    return df


# -----------------------------
# Main
# -----------------------------
def run(cfg: RunConfig) -> int:
    # 1) Fetch new
    new_df = _fetch_new(ARXIV_RSS_URLS, cfg.days_back)

    # 2) Read old
    old_df = _safe_read_parquet(LIVE_DATASET)

    # 3) Ensure schema
    new_df = _ensure_columns(new_df, strict=cfg.strict)
    old_df = _ensure_columns(old_df, strict=False) if not old_df.empty else pd.DataFrame(columns=REQUIRED_COLS)

    # 4) Normalize & merge
    new_df = _normalize_and_build_text(new_df)
    old_df = _normalize_and_build_text(old_df) if not old_df.empty else old_df

    merged = pd.concat([old_df, new_df], ignore_index=True) if not old_df.empty else new_df.copy()
    merged = _ensure_columns(merged, strict=cfg.strict)
    merged = _normalize_and_build_text(merged)
    merged = _dedup(merged)
    merged = _trim(merged, cfg.max_keep)

    # 5) Save parquet
    merged.to_parquet(LIVE_DATASET, index=False)

    # 6) Meta
    if cfg.write_meta:
        meta = {
            "updated_at_utc": _now_utc_iso(),
            "days_back": int(cfg.days_back),
            "sources": list(ARXIV_RSS_URLS),
            "rows": int(len(merged)),
            "min_date": str(merged["date"].min()) if len(merged) > 0 else None,
            "max_date": str(merged["date"].max()) if len(merged) > 0 else None,
            "parquet": str(LIVE_DATASET),
        }
        _write_meta(LIVE_META, meta)

    # 7) Output
    if len(new_df) < cfg.min_rows_warn:
        print(f"[live_runner] WARNING: muy pocos registros nuevos en esta corrida: {len(new_df)} (days_back={cfg.days_back})")

    _print_summary(merged, LIVE_DATASET, LIVE_META if cfg.write_meta else None)
    return 0


def _parse_args(argv: list[str]) -> RunConfig:
    """
    Uso:
      python -m src.live_runner
      python -m src.live_runner 14
      python -m src.live_runner 14 20000
      python -m src.live_runner 14 20000 --no-meta
      python -m src.live_runner 14 20000 --strict
    """
    days_back = 30
    max_keep = 20000
    write_meta = True
    strict = False

    # posicionales
    if len(argv) >= 2:
        try:
            days_back = int(argv[1])
        except Exception:
            pass

    if len(argv) >= 3:
        try:
            max_keep = int(argv[2])
        except Exception:
            pass

    # flags
    if "--no-meta" in argv:
        write_meta = False
    if "--strict" in argv:
        strict = True

    return RunConfig(days_back=days_back, max_keep=max_keep, write_meta=write_meta, strict=strict)


if __name__ == "__main__":
    cfg = _parse_args(sys.argv)
    raise SystemExit(run(cfg))
