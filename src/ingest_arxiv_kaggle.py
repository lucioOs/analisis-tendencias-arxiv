from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from email.utils import parsedate_to_datetime


DEFAULT_CAT_PREFIXES = (
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
)


def parse_any_date(rec: dict[str, Any]) -> Optional[pd.Timestamp]:
    versions = rec.get("versions") or []
    if isinstance(versions, list) and versions:
        v0 = versions[0] if isinstance(versions[0], dict) else {}
        created = v0.get("created")
        if created:
            try:
                dt = parsedate_to_datetime(created)
                return pd.Timestamp(dt).tz_convert(None) if dt.tzinfo else pd.Timestamp(dt)
            except Exception:
                pass

    upd = rec.get("update_date")
    if upd:
        try:
            return pd.to_datetime(upd, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            pass

    cre = rec.get("created")
    if cre:
        try:
            return pd.to_datetime(cre, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            pass

    return None


def open_text_maybe_gz(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def norm_categories(cat: Any) -> str:
    if cat is None:
        return ""
    if isinstance(cat, (list, tuple, set)):
        return " ".join(str(x) for x in cat if x is not None)
    return str(cat)


def categories_match(cat_str: str, prefixes: Sequence[str]) -> bool:
    if not prefixes:
        return True
    return any(p in cat_str for p in prefixes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/arxiv-metadata-oai-snapshot.json",
                    help="Ruta al .json o .json.gz (JSONL) de Kaggle")
    ap.add_argument("--out", default="data/processed/dataset.parquet",
                    help="Salida parquet compatible con tu pipeline")
    ap.add_argument("--min_text_len", type=int, default=50,
                    help="Longitud mínima del texto (title+abstract)")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = sin límite; >0 para pruebas rápidas")
    ap.add_argument("--since", default=None, help="Fecha mínima (YYYY-MM-DD)")
    ap.add_argument("--until", default=None, help="Fecha máxima (YYYY-MM-DD)")
    ap.add_argument("--cat-prefix", action="append", default=None,
                    help="Prefijo de categoría a incluir (repetible). Ej: --cat-prefix cs.AI")
    ap.add_argument("--use-default-cats", action="store_true",
                    help="Usa lista default de categorías de computación/IA")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No existe el archivo: {input_path}")
        return 2

    since_ts = pd.to_datetime(args.since) if args.since else None
    until_ts = pd.to_datetime(args.until) if args.until else None

    prefixes = []
    if args.use_default_cats:
        prefixes.extend(DEFAULT_CAT_PREFIXES)
    if args.cat_prefix:
        prefixes.extend(args.cat_prefix)

    rows: list[dict[str, Any]] = []
    bad_json = 0
    no_date = 0
    short_text = 0
    cat_skipped = 0
    date_skipped = 0

    with open_text_maybe_gz(input_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            # Categorías (filtrar primero para ahorrar)
            cat_str = norm_categories(rec.get("categories"))
            if prefixes and not categories_match(cat_str, prefixes):
                cat_skipped += 1
                continue

            title = str(rec.get("title") or "").strip()
            abstract = str(rec.get("abstract") or "").strip()
            text = (title + ". " + abstract).strip()

            if len(text) < int(args.min_text_len):
                short_text += 1
                continue

            dt = parse_any_date(rec)
            if dt is None or pd.isna(dt):
                no_date += 1
                continue

            # Filtro por fechas
            if since_ts is not None and dt < since_ts:
                date_skipped += 1
                continue
            if until_ts is not None and dt > until_ts:
                date_skipped += 1
                continue

            rows.append(
                {
                    "date": dt,
                    "text": text,
                    "source": "arxiv_kaggle",
                    "id": rec.get("id"),
                    "categories": rec.get("categories"),
                }
            )

            if i % 100_000 == 0:
                print(
                    f"[INFO] Leídas {i:,} | válidas {len(rows):,} | "
                    f"cat_skip {cat_skipped:,} | date_skip {date_skipped:,} | "
                    f"sin_fecha {no_date:,} | json_malos {bad_json:,}"
                )

            if args.limit and len(rows) >= int(args.limit):
                break

    if not rows:
        print("[ERROR] No se generó ninguna fila válida.")
        print(f"  - json malos: {bad_json}")
        print(f"  - sin fecha: {no_date}")
        print(f"  - texto corto: {short_text}")
        print(f"  - filtradas por categoría: {cat_skipped}")
        print(f"  - filtradas por fecha: {date_skipped}")
        return 3

    df = pd.DataFrame(rows)

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["date", "text"], keep="last")

    df = df.dropna(subset=["date", "text"]).sort_values("date").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    print(
        f"[OK] Exportado: {out_path} | rows={len(df):,} | "
        f"min={df['date'].min()} | max={df['date'].max()} | "
        f"cat_prefixes={prefixes if prefixes else 'NONE'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
