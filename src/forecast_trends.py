# src/forecast_trends.py
# Modulo de Tendencias: Series + Clasificacion + Prediccion (Historico / Batch)
# ---------------------------------------------------------------------------
# Entradas:
#   data/processed/dataset.parquet  (columnas: date, text)
# Salidas:
#   data/processed/trends_full.parquet        (period, term, count, rel_freq, growth)
#   data/processed/trend_classes.parquet      (term, class, vol_recent, growth_recent, slope_recent, priority_score, last_period, n_periods)
#   data/processed/trends_forecast.parquet    (term, period, rel_freq_pred, model)
#
# Enfoque:
# - Construccion eficiente de series por periodo usando matrices dispersas
# - Metricas robustas con periodos ORDENADOS
# - Clasificacion data-driven (percentiles)
# - Prediccion (ETS/Naive) con fallback seguro

# src/forecast_trends.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS


SPANISH_STOP_MINI = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
    "una","su","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros",
}
STOPWORDS_ALL = set(ENGLISH_STOP_WORDS).union(SPANISH_STOP_MINI)


@dataclass
class ForecastCfg:
    h: int
    seasonal_periods: int


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "text"])

    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in ["date", "created", "published", "published_at", "updated", "update_date"] if k in cols), None)
    text_col = next((cols[k] for k in ["text", "content", "summary", "abstract", "title"] if k in cols), None)

    if date_col is None or text_col is None:
        return pd.DataFrame(columns=["date", "text"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["text"] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 30]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _safe_sample_texts(df: pd.DataFrame, n: int, seed: int) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=str)
    if n <= 0 or n >= len(df):
        return df["text"]
    return df.sample(n=n, random_state=seed)["text"]


def _build_vocab_tfidf(
    texts: Iterable[str],
    top_k: int,
    min_df: int,
    ngram_max: int,
) -> np.ndarray:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS_ALL),
        max_features=int(top_k),
        min_df=int(min_df),
        ngram_range=(1, int(ngram_max)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
    )
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()

    # filtro extra: evita cosas muy cortas o numéricas
    good = [t for t in terms if len(t) >= 4 and not t.isdigit()]
    return np.array(good, dtype=object)


def _forecast_fallback(y: np.ndarray, h: int, sp: int) -> tuple[np.ndarray, str]:
    """
    Predicción batch simple, estable y sin dependencias:
    - tendencia lineal
    - + un toque estacional naive si hay historial suficiente
    """
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    n = len(y)
    if n == 0:
        return np.zeros(h, dtype=float), "fallback-vacio"
    x = np.arange(n, dtype=float)
    if n >= 2:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, float(y[-1])

    base = a * np.arange(n, n + h, dtype=float) + b

    sp = int(sp or 0)
    if sp >= 2 and n >= sp:
        season = y[-sp:]
        season_rep = np.resize(season, h)
        pred = 0.75 * base + 0.25 * season_rep
        return pred, "lineal+seasonal_naive"

    return base, "lineal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/dataset_focus.parquet")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])
    ap.add_argument("--top_k_terms", type=int, default=1200)
    ap.add_argument("--min_df", type=int, default=30)
    ap.add_argument("--ngram_max", type=int, default=3)

    # para que no muera por RAM/tiempo:
    ap.add_argument("--sample_docs", type=int, default=200_000, help="muestra para TF-IDF (vocabulario). 0 = todo")
    ap.add_argument("--random_state", type=int, default=7)

    # forecast batch:
    ap.add_argument("--forecast_h", type=int, default=6)
    ap.add_argument("--seasonal_periods", type=int, default=12)

    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_trends = out_dir / "trends_full.parquet"
    out_classes = out_dir / "trend_classes.parquet"
    out_forecast = out_dir / "trends_forecast.parquet"

    if not in_path.exists():
        print(f"[ERROR] No existe: {in_path}")
        return 2

    print(f"[INFO] Leyendo: {in_path}")
    df = pd.read_parquet(in_path)
    df = _normalize_df(df)

    if df.empty:
        print("[ERROR] Dataset vacío tras normalización.")
        return 3

    print(f"[INFO] Registros válidos: {len(df):,}")
    df["period"] = df["date"].dt.to_period(args.freq).astype(str)

    total_by_period = df.groupby("period").size().sort_index()
    periods = total_by_period.index.tolist()
    if len(periods) < 2:
        print("[ERROR] Se requieren al menos 2 periodos para tendencias.")
        return 4

    # 1) vocabulario por TF-IDF (en muestra)
    sample_texts = _safe_sample_texts(df, n=int(args.sample_docs), seed=int(args.random_state))
    print(f"[INFO] TF-IDF en muestra: {len(sample_texts):,} docs | top_k={args.top_k_terms} | min_df={args.min_df} | ngram<= {args.ngram_max}")
    terms = _build_vocab_tfidf(sample_texts, args.top_k_terms, args.min_df, args.ngram_max)

    if terms.size == 0:
        print("[ERROR] No se obtuvieron términos. Baja min_df o sube sample_docs.")
        return 5

    print(f"[INFO] Vocabulario final: {len(terms):,} términos")

    # 2) conteos por periodo con CountVectorizer fijo (rápido vs loop+contains)
    cv = CountVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS_ALL),
        vocabulary={t: i for i, t in enumerate(terms)},
        ngram_range=(1, int(args.ngram_max)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
        binary=True,  # presencia por doc
    )

    print("[INFO] Transformando textos a matriz dispersa (esto puede tardar un poco, pero es lo correcto)...")
    X = cv.transform(df["text"].values)  # (n_docs x n_terms)

    # agregación por periodo
    print("[INFO] Agregando por periodo...")
    period_codes = pd.Categorical(df["period"], categories=periods, ordered=True).codes

    counts_per_period = np.zeros((len(periods), len(terms)), dtype=np.int32)

    # loop por periodos (normalmente pocos: meses/semanas), cada sum es muy eficiente en sparse
    for pi, p in enumerate(periods):
        mask = (period_codes == pi)
        if not np.any(mask):
            continue
        v = X[mask].sum(axis=0)  # 1 x n_terms
        counts_per_period[pi, :] = np.asarray(v).ravel().astype(np.int32)

    rel_freq = counts_per_period / total_by_period.values.reshape(-1, 1)

    # trends_full long
    print("[INFO] Construyendo trends_full...")
    df_trends = pd.DataFrame(rel_freq, index=periods, columns=terms)
    df_counts = pd.DataFrame(counts_per_period, index=periods, columns=terms)

    df_long = (
        df_trends.stack()
        .rename("rel_freq")
        .reset_index()
        .rename(columns={"level_0": "period", "level_1": "term"})
    )
    df_long["count"] = df_counts.stack().values

    # 3) métricas + clasificación
    print("[INFO] Clasificando tendencias...")
    class_rows = []
    x = np.arange(len(periods), dtype=float)

    for t in terms:
        y = df_trends[t].astype(float).values
        c = int(df_counts[t].sum())
        if c < 20:  # baja señal
            continue

        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0

        first3 = float(np.mean(y[:3])) if len(y) >= 3 else float(y[0])
        last3 = float(np.mean(y[-3:])) if len(y) >= 3 else float(y[-1])
        growth = float((last3 - first3) / (first3 + 1e-9))

        mean = float(np.mean(y))
        std = float(np.std(y))
        stability = float(max(0.0, 1.0 - (std / (mean + 1e-9))))

        # umbrales más razonables para rel_freq (ajustables)
        label = "otros"
        if slope > 2e-6 and growth > 0.30:
            label = "emergente"
        elif slope < -2e-6 and growth < -0.20:
            label = "declive"
        else:
            # consolidada = alta presencia + estable + slope pequeño
            if c >= int(np.quantile(df_counts.sum(axis=0).values, 0.70)) and stability >= 0.60 and abs(slope) < 2e-6:
                label = "consolidada"

        class_rows.append(
            {"term": t, "class": label, "slope": slope, "growth": growth, "stability": stability, "total_count": c}
        )

    df_classes = pd.DataFrame(class_rows)
    if df_classes.empty:
        print("[WARN] No se pudo clasificar ningún término con los filtros actuales.")
    else:
        print(df_classes["class"].value_counts())

    # 4) forecast batch (por término, usando rel_freq)
    print("[INFO] Generando predicción batch...")
    fc_cfg = ForecastCfg(h=int(args.forecast_h), seasonal_periods=int(args.seasonal_periods))
    future_periods = [f"futuro+{i}" for i in range(1, fc_cfg.h + 1)]

    fc_rows = []
    for _, row in df_classes.sort_values(["class", "total_count"], ascending=[True, False]).iterrows():
        t = row["term"]
        y = df_trends[t].astype(float).values
        pred, model_name = _forecast_fallback(y, h=fc_cfg.h, sp=fc_cfg.seasonal_periods)
        for p, v in zip(future_periods, pred):
            fc_rows.append({"term": t, "period": p, "rel_freq_pred": float(v), "model": model_name})

    df_fc = pd.DataFrame(fc_rows)

    # 5) guardado
    print(f"[OK] Guardando: {out_trends}")
    df_long.to_parquet(out_trends, index=False)

    print(f"[OK] Guardando: {out_classes}")
    df_classes.to_parquet(out_classes, index=False)

    print(f"[OK] Guardando: {out_forecast}")
    df_fc.to_parquet(out_forecast, index=False)

    print("[OK] Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
