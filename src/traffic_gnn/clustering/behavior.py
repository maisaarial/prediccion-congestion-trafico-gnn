from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans


def calcular_pivote_cl_comp(
    trafico: pd.DataFrame,
    id_col: str = "id",
    temporal_col: str = "tipo_dia_FH",
    value_col: str = "congestion",
) -> pd.DataFrame:
    pivote = (
        trafico.groupby([id_col, temporal_col])[value_col]
        .mean()
        .reset_index()
        .pivot(index=id_col, columns=temporal_col, values=value_col)
        .sort_index()
    )
    return pivote


def generar_cluster_comportamiento(
    pivote_cl_comp: pd.DataFrame,
    n_clusters: int = 2,
    random_state: int = 42,
    column_name: str = "cluster_comportamiento",
) -> pd.DataFrame:
    df = pivote_cl_comp.copy()
    df_fit = df.dropna().copy()

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(df_fit)

    df_fit[column_name] = labels
    return df_fit
