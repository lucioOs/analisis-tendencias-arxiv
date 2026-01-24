# src/ingest_arxiv_kaggle.py
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from email.utils import parsedate_to_datetime


def parse_any_date(rec: dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Intenta obtener una fecha válida del registro Kaggle arXiv:
    - versions[0].created (formato tipo: "Mon, 2 Apr 2007 19:18:42 GMT")
    - update_date (formato tipo: "2007-05-23")
    - created (si existiera)
    """
    # 1) versions[0]["created"]
    versions = rec.get("versions") or []
    if isinstance(versions, list) and versions:
        v0 = versions[0] if isinstance(versions[0], dict) else {}
        created = v0.get("created")
        if created:
            try:
                dt = parsedate_to_datetime(created)  # maneja GMT y formatos RFC822
                return pd.Timestamp(dt).tz_convert(None) if dt.tzinfo else pd.Timestamp(dt)
            except Exception:
                pass

    # 2) update_date (YYYY-MM-DD)
    upd = rec.get("update_date")
    if upd:
        try:
            return pd.to_datetime(upd, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            pass

    # 3) created (fallback raro)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/arxiv-metadata-oai-snapshot.json",
                    help="Ruta al .json o .json.gz de Kaggle")
    ap.add_argument("--out", default="data/processed/dataset.parquet",
                    help="Salida parquet compatible con tu pipeline")
    ap.add_argument("--min_text_len", type=int, default=50, help="Longitud mínima del texto (title+abstract)")
    ap.add_argument("--limit", type=int, default=0, help="0 = sin límite; >0 para pruebas rápidas")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No existe el archivo: {input_path}")
        return 2

    rows: list[dict[str, Any]] = []
    bad_json = 0
    no_date = 0
    short_text = 0

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
                print(f"[INFO] Leídas {i:,} líneas | válidas {len(rows):,} | sin fecha {no_date:,} | json malos {bad_json:,}")

            if args.limit and len(rows) >= int(args.limit):
                break

    if not rows:
        print("[ERROR] No se generó ninguna fila válida.")
        print(f"  - json malos: {bad_json}")
        print(f"  - sin fecha: {no_date}")
        print(f"  - texto corto: {short_text}")
        print("  Revisa que el archivo sea el arxiv-metadata-oai-snapshot (JSONL) y que esté completo.")
        return 3

    df = pd.DataFrame(rows)

    # Dedup robusto
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["date", "text"], keep="last")

    df = df.dropna(subset=["date", "text"]).sort_values("date").reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    print(f"[OK] Exportado: {out_path} | rows={len(df):,} | min={df['date'].min()} | max={df['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
