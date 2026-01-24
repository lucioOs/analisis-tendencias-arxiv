# src/plotting/charts.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def tick_step(n: int, target_ticks: int = 10) -> int:
    """
    Calcula cada cuántos puntos mostrar una etiqueta en X para no saturar el eje.
    """
    n = int(n or 0)
    if n <= 0:
        return 1
    target_ticks = max(2, int(target_ticks))
    step = int(np.ceil(n / target_ticks))
    return max(1, step)


def plot_trend_periods(periods: list[str], y: np.ndarray, title: str = ""):
    """
    Grafica una serie temporal simple por periodos.
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(x, y)
    ax.set_title(title or "Tendencia por periodos")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Apariciones")
    ax.grid(True, alpha=0.25)

    step = tick_step(len(periods))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=30, ha="right")

    fig.tight_layout()
    return fig


def _linear_forecast(y: np.ndarray, h: int) -> np.ndarray:
    """
    Forecast robusto: regresión lineal con clipping a >= 0.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        last = float(y[-1]) if n else 0.0
        out = np.full(h, last, dtype=float)
        return np.clip(out, 0.0, None)

    x = np.arange(n, dtype=float)
    # Ajuste lineal y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    xf = np.arange(n, n + h, dtype=float)
    yf = a * xf + b
    return np.clip(yf, 0.0, None)


def _seasonal_repeat(y: np.ndarray, h: int, sp: int) -> np.ndarray:
    """
    Repite patrón de longitud sp (si sp>0 y hay suficientes datos).
    """
    y = np.asarray(y, dtype=float)
    if sp <= 0 or len(y) < sp:
        return _linear_forecast(y, h)

    base = y[-sp:].copy()
    reps = int(np.ceil(h / sp))
    out = np.tile(base, reps)[:h]
    return np.clip(out, 0.0, None)


def plot_forecast(
    periods: list[str],
    y: np.ndarray,
    h: int = 6,
    sp: int = 0,
    title: str = "",
) -> Tuple[plt.Figure, str]:
    """
    Grafica histórico + pronóstico.
    - Si sp>0, repite patrón estacional de longitud sp.
    - Si no, usa una tendencia lineal.
    """
    y = np.asarray(y, dtype=float)
    h = max(1, int(h))
    sp = max(0, int(sp))

    if sp > 0:
        yhat = _seasonal_repeat(y, h, sp)
        model_name = f"Patrón repetitivo (sp={sp})"
    else:
        yhat = _linear_forecast(y, h)
        model_name = "Tendencia lineal"

    x_hist = np.arange(len(periods))
    x_fut = np.arange(len(periods), len(periods) + h)

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(x_hist, y, label="Histórico")
    ax.plot(x_fut, yhat, linestyle="--", label="Pronóstico")
    ax.set_title(title or "Predicción")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Apariciones")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Etiquetas de X
    full_periods = list(periods) + [f"+{i+1}" for i in range(h)]
    step = tick_step(len(full_periods))
    ax.set_xticks(np.arange(len(full_periods))[::step])
    ax.set_xticklabels([full_periods[i] for i in range(0, len(full_periods), step)], rotation=30, ha="right")

    fig.tight_layout()
    return fig, model_name
