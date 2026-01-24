from __future__ import annotations

import argparse
import json
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

import pandas as pd


# -----------------------------
# Logging profesional (simple)
# -----------------------------
def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log("ERROR", msg)
    raise SystemExit(code)


# -----------------------------
# Configuración
# -----------------------------
@dataclass(frozen=True)
class ProjectPaths:
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)


TEXT_KEYS = [
    "title", "headline", "heading",
    "content", "article", "body", "text",
    "summary", "description", "abstract"
]

DATE_KEYS = [
    "date", "published", "published_at", "publish_date",
    "time", "datetime", "created", "created_at",
    "year"
]


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def find_best_column(columns: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    """
    Selecciona columna por:
    1) match exacto normalizado
    2) contains match (key dentro del nombre)
    """
    norm = {normalize_name(c): c for c in columns}
    cols_norm = list(norm.keys())

    for k in keys:
        kn = normalize_name(k)
        if kn in norm:
            return norm[kn]

    for c in cols_norm:
        for k in keys:
            if normalize_name(k) in c:
                return norm[c]

    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"No existe el archivo: {path}")

    encs = ["utf-8", "utf-8-sig", "latin-1"]
    last = None
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc)
            log("INFO", f"CSV leído con encoding={enc} | shape={df.shape}")
            return df
        except Exception as e:
            last = e

    die(f"No pude leer el CSV {path.name}. Último error: {last}")
    return pd.DataFrame()  # unreachable


def parse_dates(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convierte fechas de forma robusta:
    - si la columna parece 'year', construye YYYY-01-01
    - intenta parseo general
    - devuelve datetime naive (para tu pipeline), pero consistente
    """
    s = series.copy()

    if "year" in normalize_name(col_name):
        s = s.astype(str).str.extract(r"(\d{4})")[0]
        s = pd.to_datetime(s + "-01-01", errors="coerce")
        return s

    # parseo general (a veces viene con zona horaria)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    # convertir a naive para evitar problemas en parquet/periodos
    return dt.dt.tz_convert(None)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Carga robusta del dataset MIT AI News")
    ap.add_argument("--file", default="mit_ai_news.csv", help="Nombre del CSV dentro de data/raw")
    ap.add_argument("--text_col", default="", help="Forzar nombre de columna de texto (opcional)")
    ap.add_argument("--date_col", default="", help="Forzar nombre de columna de fecha (opcional)")
    ap.add_argument("--title_col", default="", help="Forzar nombre de columna de título (opcional)")
    ap.add_argument("--drop_duplicates", action="store_true", help="Eliminar duplicados por (text,date)")
    ap.add_argument("--min_text_len", type=int, default=20, help="Longitud mínima del texto final")
    args = ap.parse_args()

    P = ProjectPaths()
    P.ensure()

    src_path = P.raw / args.file
    df = safe_read_csv(src_path)

    cols = list(df.columns)
    log("INFO", f"Columnas detectadas: {cols}")

    # Determinar columnas
    date_col = args.date_col.strip() or find_best_column(cols, DATE_KEYS)
    if not date_col:
        die("No se encontró columna de FECHA. Usa --date_col para forzarla.")

    # Texto: idealmente combinar title+content si existen
    title_col = args.title_col.strip() or find_best_column(cols, ["title", "headline", "heading"])
    text_col = args.text_col.strip() or find_best_column(cols, ["content", "body", "text", "article", "summary", "description"])

    if not title_col and not text_col:
        die("No se encontró columna de TEXTO. Usa --text_col/--title_col para forzar.")

    log("INFO", f"Fecha: {date_col}")
    log("INFO", f"Título: {title_col if title_col else '(no detectado)'}")
    log("INFO", f"Cuerpo:  {text_col if text_col else '(no detectado)'}")

    # Construir texto final
    out = pd.DataFrame()
    if title_col and text_col and title_col != text_col:
        out["text"] = (
            df[title_col].astype(str).fillna("").str.strip()
            + ". "
            + df[text_col].astype(str).fillna("").str.strip()
        ).str.strip()
    elif text_col:
        out["text"] = df[text_col].astype(str).fillna("").str.strip()
    else:
        out["text"] = df[title_col].astype(str).fillna("").str.strip()

    out["source"] = "mit_ai_news"
    out["date"] = parse_dates(df[date_col], date_col)

    # Limpieza básica y validaciones
    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    before = len(out)
    out = out.dropna(subset=["date"])
    out = out[out["text"].str.len() >= args.min_text_len].copy()
    after = len(out)

    if out.empty:
        die(
            "Tras limpieza, el dataset quedó vacío. "
            "Revisa columnas reales o baja --min_text_len."
        )

    if args.drop_duplicates:
        out = out.drop_duplicates(subset=["text", "date"]).copy()

    # Guardar
    out_path = P.processed / "dataset.parquet"
    out.to_parquet(out_path, index=False)

    # Guardar metadata profesional (para tu tesis)
    meta = {
        "dataset": "MIT AI News",
        "file": src_path.name,
        "rows_input": int(before),
        "rows_output": int(len(out)),
        "columns_input": cols,
        "selected": {
            "date_col": date_col,
            "title_col": title_col,
            "text_col": text_col,
            "combined_title_body": bool(title_col and text_col and title_col != text_col),
        },
        "quality": {
            "null_date_rate_output": float(out["date"].isna().mean()),
            "min_text_len": args.min_text_len,
        },
        "date_range": {
            "min": str(out["date"].min()),
            "max": str(out["date"].max()),
        },
    }
    write_json(P.processed / "dataset_schema.json", meta)

    log("INFO", f"✅ dataset.parquet creado: {out_path}")
    log("INFO", f"Filas salida: {len(out)}")
    log("INFO", f"Rango fechas: {out['date'].min()} -> {out['date'].max()}")
    log("INFO", f"Metadata: {P.processed / 'dataset_schema.json'}")


if __name__ == "__main__":
    main()
