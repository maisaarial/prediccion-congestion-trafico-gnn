from __future__ import annotations

import pandas as pd


SPANISH_WEEKDAY = {
    0: "lunes",
    1: "martes",
    2: "miercoles",
    3: "jueves",
    4: "viernes",
    5: "sabado",
    6: "domingo",
}


def _build_franja_horaria(hour: int) -> str:
    start = (hour // 2) * 2
    end = start + 2
    return f"{start:02d}-{end:02d}"


def obtener_variables_temporales(trafico: pd.DataFrame, fecha_col: str = "fecha") -> pd.DataFrame:
    df = trafico.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
    df = df.dropna(subset=[fecha_col]).copy()

    df["anio"] = df[fecha_col].dt.year
    df["mes"] = df[fecha_col].dt.month
    df["dia"] = df[fecha_col].dt.day
    df["hora"] = df[fecha_col].dt.hour
    df["minute"] = df[fecha_col].dt.minute
    df["dia_semana_num"] = df[fecha_col].dt.dayofweek
    df["dia_semana"] = df["dia_semana_num"].map(SPANISH_WEEKDAY)
    df["tipo_dia"] = df["dia_semana_num"].apply(lambda x: "L" if x < 5 else "F")
    df["FH"] = df["hora"].apply(_build_franja_horaria)
    df["tipo_dia_FH"] = df["tipo_dia"] + " " + df["FH"]
    return df
