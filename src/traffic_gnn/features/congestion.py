from __future__ import annotations

import numpy as np
import pandas as pd


def calcular_congestion(
    trafico: pd.DataFrame,
    intensidad_col: str = "intensidad",
    ocupacion_col: str = "ocupacion",
    method: str = "ocupacion_sobre_intensidad",
    clip_percentile: float | None = 99.0,
) -> pd.DataFrame:
    """
    Calcula una métrica continua de congestión.

    Notas:
    - Los valores negativos se interpretan como faltantes.
    - La fórmula puede ajustarse después según la definición final del TFM.
    """
    df = trafico.copy()

    for col in [intensidad_col, ocupacion_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    intensidad = df[intensidad_col].astype(float)
    ocupacion = df[ocupacion_col].astype(float)

    if method == "ocupacion_sobre_intensidad":
        denominador = intensidad.replace(0, np.nan)
        congestion = ocupacion / denominador
    elif method == "ocupacion_por_vehiculos_intervalo":
        vehiculos_intervalo = (intensidad / 4.0).replace(0, np.nan)
        congestion = ocupacion / vehiculos_intervalo
    else:
        raise ValueError(f"Método no soportado: {method}")

    if clip_percentile is not None:
        finite = congestion[np.isfinite(congestion)]
        if len(finite) > 0:
            upper = np.nanpercentile(finite, clip_percentile)
            congestion = congestion.clip(upper=upper)

    df["congestion"] = congestion.fillna(0.0)
    return df
