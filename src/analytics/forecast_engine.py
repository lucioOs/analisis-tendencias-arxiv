from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _try_ets_forecast(y: np.ndarray, h: int, seasonal_periods: int) -> Optional[np.ndarray]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception:
        return None

    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return None

    sp = int(seasonal_periods or 0)
    use_seasonal = sp >= 2 and len(y) >= (2 * sp)

    trend = "add" if len(y) >= 6 else None
    seasonal = "add" if use_seasonal else None

    try:
        model = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=sp if use_seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        pred = fit.forecast(h)
        return np.asarray(pred, dtype=float)
    except Exception:
        return None


def _fallback_forecast(y: np.ndarray, h: int, seasonal_periods: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(h, dtype=float)

    x = np.arange(n, dtype=float)
    if n >= 2:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, float(y[-1])

    base = a * (np.arange(n, n + h, dtype=float)) + b

    sp = int(seasonal_periods or 0)
    if sp >= 2 and n >= sp:
        season = y[-sp:]
        season_rep = np.resize(season, h)
        return 0.75 * base + 0.25 * season_rep

    return base


def forecast_series(y: np.ndarray, h: int = 6, seasonal_periods: int = 0) -> Tuple[np.ndarray, str]:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    pred = _try_ets_forecast(y, h=h, seasonal_periods=seasonal_periods)
    if pred is not None:
        return pred, "Modelo estadístico"

    return _fallback_forecast(y, h=h, seasonal_periods=seasonal_periods), "Modelo simple"
