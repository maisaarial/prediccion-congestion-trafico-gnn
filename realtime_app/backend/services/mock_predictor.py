from __future__ import annotations

import pandas as pd


PREDICTOR_MODE = "DEMO / SIN MODELO REAL"


def predict_next_15min(current_entities: pd.DataFrame) -> pd.DataFrame:
    out = current_entities.copy()
    current = pd.to_numeric(out["congestion_actual"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["prediccion_15min"] = current
    out["diferencia"] = out["prediccion_15min"] - current
    out["predictor"] = "mock_identity"
    out["modo_modelo"] = "demo_sin_modelo_real"
    return out
