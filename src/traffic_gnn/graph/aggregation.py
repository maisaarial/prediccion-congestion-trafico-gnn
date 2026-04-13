from __future__ import annotations

import pandas as pd


def aggregation_congestion_por_clusters(
    dataframe_congestion_cluster: pd.DataFrame,
    nombre_col_cluster: str,
    fecha_col: str = "fecha",
    value_col: str = "congestion",
) -> pd.DataFrame:
    df_avg = (
        dataframe_congestion_cluster.groupby([nombre_col_cluster, fecha_col])[value_col]
        .mean()
        .reset_index()
        .rename(columns={nombre_col_cluster: "cluster"})
    )
    return df_avg


def calcular_centroides_clusters(
    info_centroides_cl: pd.DataFrame,
    cluster_col: str,
    x_col: str = "utm_x",
    y_col: str = "utm_y",
) -> pd.DataFrame:
    centroides = (
        info_centroides_cl.groupby(cluster_col)[[x_col, y_col]]
        .mean()
        .reset_index()
        .rename(columns={cluster_col: "cluster"})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    return centroides
