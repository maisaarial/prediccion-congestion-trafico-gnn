from __future__ import annotations

import pandas as pd


def intersectar_clusters(
    cluster_proximidad: pd.DataFrame,
    cluster_comportamiento: pd.DataFrame,
    id_col: str = "id",
    prox_col: str = "cluster_proximidad",
    behavior_col: str = "cluster_comportamiento",
    output_col: str = "cluster_interseccion",
) -> pd.DataFrame:
    df = cluster_proximidad.merge(cluster_comportamiento, on=id_col, how="inner")
    pares = list(zip(df[prox_col], df[behavior_col]))
    df[output_col] = pd.factorize(pares)[0]
    return df


def intersectar_clusters_sentido_v1(
    cluster_proximidad: pd.DataFrame,
    cluster_sentido: pd.DataFrame,
    id_col: str = "id",
    prox_col: str = "cluster_proximidad",
    sentido_col: str = "sentido_v1",
    output_col: str = "cluster_prox_sentido_v1",
) -> pd.DataFrame:
    df = cluster_proximidad.merge(cluster_sentido[[id_col, sentido_col]], on=id_col, how="inner")
    df[output_col] = pd.factorize(list(zip(df[prox_col], df[sentido_col])))[0]
    return df


def intersectar_clusters_sentido_v2(
    cluster_proximidad: pd.DataFrame,
    cluster_sentido: pd.DataFrame,
    id_col: str = "id",
    prox_col: str = "cluster_proximidad",
    sentido_col: str = "sentido_v2",
    output_col: str = "cluster_prox_sentido_v2",
) -> pd.DataFrame:
    df = cluster_proximidad.merge(cluster_sentido[[id_col, sentido_col]], on=id_col, how="inner")
    df[output_col] = pd.factorize(list(zip(df[prox_col], df[sentido_col])))[0]
    return df
