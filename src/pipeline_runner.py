from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REQ_FILES = [
    Path("data/processed/dataset.parquet"),
    Path("data/processed/clean.parquet"),
    Path("data/processed/trends_full.parquet"),
    Path("data/processed/trend_classes.parquet"),
    Path("models/model.pkl"),
]


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def exists_all(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


def main() -> None:
    # Ejecuta el pipeline completo si faltan outputs
    if exists_all(REQ_FILES):
        print("[INFO] Outputs ya existen. No se ejecuta pipeline.", flush=True)
        return

    print("[INFO] Faltan outputs. Ejecutando pipeline completo...", flush=True)

    # 1) Loader (MIT)
    run([sys.executable, "src/load_data.py", "--file", "mit_ai_news.csv", "--drop_duplicates"])

    # 2) Preprocess
    run([sys.executable, "src/preprocess.py", "--min_len", "30"])

    # 3) Tendencias full
    run([sys.executable, "src/trends_full.py", "--freq", "M", "--min_df", "3", "--ngram_max", "2"])

    # 4) Clasificación
    run([sys.executable, "src/classify_trends.py", "--freq_window", "6", "--min_periods", "8"])

    # 5) Features + Train (para /predict)
    run([sys.executable, "src/features.py", "--horizon", "1"])
    run([sys.executable, "src/train.py", "--n_estimators", "500"])

    print("[INFO] ✅ Pipeline completo.", flush=True)


if __name__ == "__main__":
    main()
