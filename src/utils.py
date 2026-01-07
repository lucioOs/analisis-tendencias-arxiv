from __future__ import annotations

import sys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("models")

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    raise SystemExit(code)


def show_head(df: pd.DataFrame, n: int = 3) -> None:
    log("Head:")
    log(df.head(n).to_string(index=False))


def normalize_colname(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"\s+", "_", c)
    return c


def find_best_column(columns: Sequence[str], keys: Iterable[str]) -> Optional[str]:
    """
    Encuentra la mejor columna:
    - exact match por nombre normalizado
    - contains match (columna contiene key)
    """
    norm_map = {normalize_colname(c): c for c in columns}
    norm_cols = list(norm_map.keys())

    keys_norm = [normalize_colname(k) for k in keys]

    # 1) exact match
    for k in keys_norm:
        if k in norm_map:
            return norm_map[k]

    # 2) contains match
    for col_norm in norm_cols:
        for k in keys_norm:
            if k in col_norm:
                return norm_map[col_norm]

    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"No existe el archivo: {path}")

    # intenta con encoding común
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    die(f"No pude leer CSV {path.name}. Último error: {last_err}")
    return pd.DataFrame()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
