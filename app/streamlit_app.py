# app/streamlit_app.py
# Dashboard de Tendencias (PLN + ML)
# - Modo Historico (MIT): tendencias completas (emergente/consolidada/declive)
# - Modo Live (RSS + arXiv): si hay historia suficiente, calcula tendencias; si no, muestra actividad (siempre muestra algo)
#
# Requisitos esperados en data/processed:
# - dataset.parquet (historico MIT)
# - trends_full.parquet / trend_classes.parquet (si ya corriste pipeline historico)
# - live_dataset.parquet (si ya corriste export_live_dataset.py)
#
# Este archivo esta pensado para ser robusto: valida entradas, maneja faltantes y evita pantallas vacias.

from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Config / Paths
# -----------------------------
ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"

HIST_DATASET = PROCESSED / "dataset.parquet"
LIVE_DATASET = PROCESSED / "live_dataset.parquet"

TRENDS_FULL = PROCESSED / "trends_full.parquet"
TREND_CLASSES = PROCESSED / "trend_classes.parquet"

# Si tienes un runner live, cambia a la ruta correcta. Si no existe, el boton solo refresca la pagina.
LIVE_RUNNER = ROOT / "src" / "live_runner.py"  # opcional
EXPORT_LIVE = ROOT / "src" / "export_live_dataset.py"  # opcional

APP_TITLE = "Sistema de Analisis y Prediccion de Tendencias (PLN + ML)"


# -----------------------------
# Utilidades (logs, safe I/O)
# -----------------------------
def _safe_text(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def log_info(msg: str) -> None:
    st.caption(_safe_text(msg))


def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def read_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        if not file_exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"No se pudo leer {path}: {e}")
        return pd.DataFrame()


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza una base para tener al menos: date, text, source.
    Si no existen, intenta inferir. Si no se puede, regresa df vacio.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}

    # Fecha
    date_col = None
    for k in ["date", "published", "published_at", "created", "updated", "time", "datetime"]:
        if k in cols:
            date_col = cols[k]
            break

    # Texto
    text_col = None
    for k in ["text", "content", "summary", "abstract", "body", "article_body", "article", "description"]:
        if k in cols:
            text_col = cols[k]
            break

    # Fuente
    source_col = None
    for k in ["source", "origin", "provider"]:
        if k in cols:
            source_col = cols[k]
            break

    if date_col is None or text_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["text"] = df[text_col].astype(str)

    if source_col is not None:
        out["source"] = df[source_col].astype(str)
    else:
        out["source"] = "unknown"

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def period_count(df: pd.DataFrame, freq: str) -> int:
    if df.empty:
        return 0
    try:
        return df["date"].dt.to_period(freq).nunique()
    except Exception:
        return 0


def date_range_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "-"
    try:
        return f"{df['date'].min()} -> {df['date'].max()}"
    except Exception:
        return "-"


def run_script_capture(args: list[str], timeout_sec: int = 120) -> Tuple[int, str]:
    """
    Ejecuta un script y regresa (returncode, stdout+stderr).
    Maneja encoding para Windows.
    """
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "Tiempo de espera excedido al ejecutar el proceso."
    except Exception as e:
        return 1, f"Fallo al ejecutar proceso: {e}"


# -----------------------------
# Tendencias / Actividad (Live fallback)
# -----------------------------
@dataclass
class LiveActivityResult:
    top_terms: pd.DataFrame
    activity_by_day: pd.DataFrame
    recent_rows: pd.DataFrame


def compute_live_activity(
    df: pd.DataFrame,
    ngram_max: int,
    min_df: int,
    top_k_terms: int = 40,
    max_features: int = 4000,
) -> LiveActivityResult:
    """
    Calcula "actividad" cuando no hay suficiente historia para tendencias:
    - Top terminos por TF-IDF (suma de pesos)
    - Actividad por dia (conteo de items)
    - Tabla de articulos recientes
    """
    if df.empty:
        return LiveActivityResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # TF-IDF robusto para terminos
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=max_features,
        ngram_range=(1, max(1, int(ngram_max))),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
        min_df=max(1, int(min_df)),
    )

    X = vectorizer.fit_transform(df["text"].values)
    terms = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    if len(scores) == 0:
        top_terms = pd.DataFrame(columns=["term", "score"])
    else:
        idx = scores.argsort()[::-1][:top_k_terms]
        top_terms = pd.DataFrame({"term": terms[idx], "score": scores[idx]})

    # Actividad por dia
    tmp = df.copy()
    tmp["day"] = tmp["date"].dt.date
    activity = tmp.groupby("day", as_index=False).size().rename(columns={"size": "items"})

    # Tabla de recientes
    recent = df.sort_values("date", ascending=False).head(30).copy()
    recent = recent[["date", "source", "text"]]

    return LiveActivityResult(top_terms, activity, recent)


# -----------------------------
# UI helpers
# -----------------------------
def apply_dark_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { letter-spacing: 0.2px; }
        .stAlert { border-radius: 14px; }
        div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 14px; }
        .stDataFrame { border-radius: 14px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.title(APP_TITLE)
    st.write("Modo historico (MIT) y modo Live (near real-time con arXiv/RSS, sin costo).")


def quick_guide() -> None:
    with st.expander("Guia rapida de interpretacion", expanded=False):
        st.markdown(
            """
- **Historico (MIT)**: base amplia (1994-2023). Permite clasificar tendencias (emergente / consolidada / declive) con mayor estabilidad.
- **Live (RSS + arXiv)**: datos recientes. Si aun no hay suficientes periodos temporales (semanas/meses), se muestra **actividad** (top terminos y articulos recientes).
- La clasificacion de tendencia requiere **al menos 2 periodos** en la frecuencia seleccionada.
            """.strip()
        )


def show_kpis(df: pd.DataFrame, label: str, freq: str) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Registros ({label})", f"{len(df):,}")
    with c2:
        st.metric("Rango de fechas", date_range_str(df))
    with c3:
        st.metric(f"Periodos ({freq})", f"{period_count(df, freq)}")


# -----------------------------
# Historico: lectura de tendencias
# -----------------------------
def load_historic_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_trends = read_parquet_safe(TRENDS_FULL)
    df_classes = read_parquet_safe(TREND_CLASSES)

    # Normaliza nombres esperados si existen
    # df_trends esperado: term, period, count, rel_freq
    # df_classes esperado: term, class (emergente/consolidada/declive)
    if not df_trends.empty:
        cols = {c.lower(): c for c in df_trends.columns}
        rename = {}
        if "term" in cols and cols["term"] != "term":
            rename[cols["term"]] = "term"
        if "period" in cols and cols["period"] != "period":
            rename[cols["period"]] = "period"
        if "count" in cols and cols["count"] != "count":
            rename[cols["count"]] = "count"
        if "rel_freq" in cols and cols["rel_freq"] != "rel_freq":
            rename[cols["rel_freq"]] = "rel_freq"
        if rename:
            df_trends = df_trends.rename(columns=rename)

    if not df_classes.empty:
        cols = {c.lower(): c for c in df_classes.columns}
        rename = {}
        if "term" in cols and cols["term"] != "term":
            rename[cols["term"]] = "term"
        if "class" in cols and cols["class"] != "class":
            rename[cols["class"]] = "class"
        if rename:
            df_classes = df_classes.rename(columns=rename)

    return df_trends, df_classes


def historic_view() -> None:
    st.subheader("Historico (MIT)")

    df_base = normalize_base(read_parquet_safe(HIST_DATASET))
    if df_base.empty:
        st.error(
            f"No se encontro base historica en {HIST_DATASET}. "
            "Ejecuta el pipeline historico primero (load_data, preprocess, trends, features, train)."
        )
        return

    # Mostrar KPIs de base
    show_kpis(df_base, "Historico", "M")

    df_trends, df_classes = load_historic_tables()

    if df_trends.empty or df_classes.empty:
        st.warning(
            "Faltan archivos de tendencias historicas (trends_full.parquet o trend_classes.parquet). "
            "Puedes seguir viendo la base, pero no se mostrara clasificacion."
        )
        # Mostrar un resumen simple por anio/mes
        tmp = df_base.copy()
        tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
        counts = tmp.groupby("month", as_index=False).size().rename(columns={"size": "items"})
        st.line_chart(counts.set_index("month")[["items"]], use_container_width=True)
        st.dataframe(df_base.sort_values("date", ascending=False).head(30), use_container_width=True)
        return

    # Unir clases con tendencias
    df = df_trends.merge(df_classes, on="term", how="left")
    df["class"] = df["class"].fillna("sin_clase")

    # Tabs por clase
    classes_order = ["emergente", "consolidada", "declive", "sin_clase"]
    tabs = st.tabs([c.capitalize() for c in classes_order])

    for cls, tab in zip(classes_order, tabs):
        with tab:
            sub = df[df["class"].str.lower() == cls].copy()
            if sub.empty:
                st.info("No hay terminos en esta categoria.")
                continue

            term_list = (
                sub.groupby("term", as_index=False)["count"].sum()
                .sort_values("count", ascending=False)
                .head(200)["term"]
                .tolist()
            )
            term = st.selectbox("Selecciona un termino", term_list, key=f"hist_term_{cls}")

            ts = sub[sub["term"] == term].copy()
            if ts.empty:
                st.info("No hay serie temporal para este termino.")
                continue

            # Grafica serie temporal
            ts = ts.sort_values("period")
            st.line_chart(ts.set_index("period")[["rel_freq"]], use_container_width=True)

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(ts[["period", "count", "rel_freq"]], use_container_width=True, height=360)
            with c2:
                st.write("Resumen")
                st.metric("Total apariciones", f"{ts['count'].sum():,}")
                st.metric("Promedio rel_freq", f"{ts['rel_freq'].mean():.6f}")


# -----------------------------
# Live view
# -----------------------------
def live_controls() -> Dict[str, object]:
    with st.sidebar:
        st.header("Live")

        auto_refresh = st.selectbox("Auto-refresh (min)", [0, 1, 5, 10, 15], index=2)
        do_update = st.button("Actualizar ahora", use_container_width=True)

        st.caption("Auto-refresh recarga la vista. 'Actualizar ahora' ejecuta el export live (si existe).")

    return {"auto_refresh": auto_refresh, "do_update": do_update}


def maybe_autorefresh(minutes: int) -> None:
    if minutes and minutes > 0:
        # Streamlit no tiene auto-refresh nativo sin dependencias externas.
        # Implementacion simple: usa query params con un contador para forzar rerun.
        # Si esto te molesta, deja auto_refresh en 0.
        now = int(time.time())
        st.query_params["t"] = str(now)


def run_live_update() -> None:
    """
    Intenta actualizar live:
    - si existe src/export_live_dataset.py, lo ejecuta
    - si no, solo avisa
    """
    if file_exists(EXPORT_LIVE):
        st.info("Actualizando base Live...")
        code, out = run_script_capture(["python", str(EXPORT_LIVE)], timeout_sec=180)
        if code == 0:
            st.success("Live actualizado. Recargando datos.")
        else:
            st.error("No se pudo actualizar Live.")
            st.text(out[:4000])
    elif file_exists(LIVE_RUNNER):
        # fallback a live_runner si tu proyecto lo usa
        st.info("Actualizando Live (runner)...")
        code, out = run_script_capture(["python", str(LIVE_RUNNER)], timeout_sec=220)
        if code == 0:
            st.success("Live actualizado. Recargando datos.")
        else:
            st.error("No se pudo actualizar Live.")
            st.text(out[:4000])
    else:
        st.warning("No hay script de actualizacion Live configurado. Se usaran los datos existentes.")


def live_view() -> None:
    st.subheader("Live (RSS + arXiv)")

    df_live = normalize_base(read_parquet_safe(LIVE_DATASET))
    if df_live.empty:
        st.warning(
            f"No se encontro {LIVE_DATASET}. Presiona 'Actualizar ahora' si tienes export_live_dataset.py, "
            "o genera live_dataset.parquet desde tu pipeline."
        )
        return

    # Controles de analisis live
    c1, c2, c3 = st.columns(3)
    with c1:
        window_days = st.slider("Ventana Live (dias)", min_value=30, max_value=365, value=365, step=5)
    with c2:
        freq = st.selectbox("Frecuencia", ["D", "W", "M"], index=1)
    with c3:
        min_df = st.slider("min_df", min_value=1, max_value=20, value=1, step=1)

    ngram_max = st.selectbox("ngram_max", [1, 2], index=0)

    # Filtrar ventana
    cutoff = df_live["date"].max() - pd.Timedelta(days=int(window_days))
    dfw = df_live[df_live["date"] >= cutoff].copy()
    if dfw.empty:
        st.info("No hay datos en la ventana seleccionada. Aumenta la ventana.")
        return

    show_kpis(dfw, "Live", freq)

    n_periods = period_count(dfw, freq)

    # Si no hay suficiente historia para tendencias: mostrar actividad SIEMPRE
    if n_periods <= 1:
        st.info(
            "Modo Live: solo se detecto un periodo temporal. "
            "Se mostrara actividad (frecuencias y articulos recientes). "
            "La clasificacion de tendencia se habilita automaticamente con 2 o mas periodos."
        )

        try:
            activity = compute_live_activity(dfw, ngram_max=int(ngram_max), min_df=int(min_df))
        except Exception as e:
            st.error(f"No se pudo calcular actividad Live: {e}")
            st.dataframe(dfw.sort_values("date", ascending=False).head(30), use_container_width=True)
            return

        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Top terminos recientes")
            if activity.top_terms.empty:
                st.info("No se encontraron terminos con los parametros actuales.")
            else:
                st.dataframe(activity.top_terms, use_container_width=True, height=420)

        with right:
            st.subheader("Actividad por dia")
            if activity.activity_by_day.empty:
                st.info("No hay suficientes puntos para graficar actividad.")
            else:
                st.line_chart(activity.activity_by_day.set_index("day")[["items"]], use_container_width=True)

        st.subheader("Articulos recientes")
        st.dataframe(activity.recent_rows, use_container_width=True, height=380)
        return

    # Si hay 2+ periodos: construir conteos por periodo y clasificar tendencia (simple y robusto)
    st.success("Se detecto historia suficiente. Se calcularan tendencias Live.")

    # Construir serie por termino/periodo usando TF-IDF para seleccionar terminos
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            max_features=5000,
            ngram_range=(1, int(ngram_max)),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
            min_df=int(min_df),
        )
        X = vec.fit_transform(dfw["text"].values)
        terms = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        if len(scores) == 0:
            st.info("No hay terminos suficientes con los parametros actuales. Baja min_df o cambia ngram_max.")
            return
        top_k = 60
        top_idx = scores.argsort()[::-1][:top_k]
        top_terms = terms[top_idx].tolist()
    except Exception as e:
        st.error(f"No se pudieron construir terminos Live: {e}")
        return

    # Tokenizacion ligera por conteo de ocurrencias: para robustez, aproximamos con presencia por documento.
    # Si quieres conteo exacto, se puede extender; este enfoque es estable y rapido.
    dfw = dfw.copy()
    dfw["period"] = dfw["date"].dt.to_period(freq).astype(str)

    # Conteo por termino/periodo (presencia por documento)
    rows = []
    text_series = dfw["text"].str.lower()

    for term in top_terms:
        # busqueda simple. Para bigramas, el termino incluye espacio
        mask = text_series.str.contains(term, regex=False)
        if mask.sum() == 0:
            continue
        tmp = dfw.loc[mask, ["period"]].copy()
        cnt = tmp.groupby("period").size().rename("count").reset_index()
        cnt["term"] = term
        rows.append(cnt)

    if not rows:
        st.info("No se generaron series. Prueba con ngram_max=1 y min_df=1.")
        return

    trend_df = pd.concat(rows, ignore_index=True)
    trend_df = trend_df.sort_values(["term", "period"]).reset_index(drop=True)

    # Completar periodos faltantes por termino con 0
    all_periods = sorted(trend_df["period"].unique().tolist())
    full = []
    for term in sorted(trend_df["term"].unique().tolist()):
        sub = trend_df[trend_df["term"] == term].set_index("period")
        sub = sub.reindex(all_periods, fill_value=0)
        sub = sub.reset_index().rename(columns={"index": "period"})
        sub["term"] = term
        full.append(sub)

    trend_df = pd.concat(full, ignore_index=True)
    trend_df["count"] = trend_df["count"].astype(int)

    # Clasificacion simple por pendiente sobre los ultimos N puntos
    # emergente: pendiente positiva marcada
    # declive: pendiente negativa marcada
    # consolidada: estable
    def classify(series: pd.Series) -> str:
        y = series.values.astype(float)
        if len(y) < 2:
            return "actividad"
        # Pendiente con regresion lineal simple
        x = pd.Series(range(len(y))).values.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return "consolidada"
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom

        # Normalizar por magnitud
        scale = max(1.0, y.max())
        slope_n = slope / scale

        if slope_n >= 0.08:
            return "emergente"
        if slope_n <= -0.08:
            return "declive"
        return "consolidada"

    classes = []
    for term in sorted(trend_df["term"].unique().tolist()):
        sub = trend_df[trend_df["term"] == term].sort_values("period")
        cls = classify(sub["count"])
        classes.append({"term": term, "class": cls})

    classes_df = pd.DataFrame(classes)
    trend_df = trend_df.merge(classes_df, on="term", how="left")

    # Tabs por clase
    classes_order = ["emergente", "consolidada", "declive"]
    tabs = st.tabs([c.capitalize() for c in classes_order])

    for cls, tab in zip(classes_order, tabs):
        with tab:
            sub = trend_df[trend_df["class"] == cls].copy()
            if sub.empty:
                st.info("No hay terminos en esta categoria con los parametros actuales.")
                continue

            # Ranking por volumen total
            ranking = (
                sub.groupby("term", as_index=False)["count"].sum()
                .sort_values("count", ascending=False)
                .head(200)
            )
            term = st.selectbox("Selecciona un termino", ranking["term"].tolist(), key=f"live_term_{cls}")

            ts = sub[sub["term"] == term].sort_values("period")
            st.line_chart(ts.set_index("period")[["count"]], use_container_width=True)

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(ts[["period", "count"]], use_container_width=True, height=360)
            with c2:
                st.write("Resumen")
                st.metric("Total apariciones (presencia)", f"{ts['count'].sum():,}")
                st.metric("Periodos", f"{len(ts):,}")


# -----------------------------
# Main App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_dark_css()

    controls = live_controls()
    auto_refresh = int(controls["auto_refresh"])
    do_update = bool(controls["do_update"])

    header()
    quick_guide()

    # Auto-refresh (opcional)
    if auto_refresh > 0:
        maybe_autorefresh(auto_refresh)

    st.markdown("### Modo de datos")
    mode = st.radio("Selecciona el modo", ["Historico (MIT)", "Live (RSS + arXiv)"], horizontal=True)

    if mode == "Live (RSS + arXiv)":
        if do_update:
            run_live_update()
            st.rerun()
        live_view()
    else:
        historic_view()


if __name__ == "__main__":
    # Asegura encoding UTF-8 en Windows cuando sea posible
    try:
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except Exception:
        pass
    main()
