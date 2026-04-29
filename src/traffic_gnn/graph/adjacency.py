from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


def adjacency_knn_from_centroids(
    centroides: pd.DataFrame,
    k: int = 5,
    x_col: str = "utm_x",
    y_col: str = "utm_y",
) -> pd.DataFrame:
    """Construye aristas kNN desde centroides ya reindexados 0..N-1."""
    centroides = centroides.sort_values("cluster").reset_index(drop=True).copy()
    coords = centroides[[x_col, y_col]].values
    n = len(coords)
    if n == 0:
        raise ValueError("No hay centroides para construir la adyacencia.")
    if n == 1:
        return pd.DataFrame({"source": [0], "target": [0], "distance": [0.0], "weight": [1.0]})

    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(coords)
    distancias, indices = nbrs.kneighbors(coords)

    edges = []
    for i in range(n):
        source = int(centroides.loc[i, "cluster"])
        for dist, j in zip(distancias[i, 1:], indices[i, 1:]):
            target = int(centroides.loc[j, "cluster"])
            edges.append((source, target, float(dist)))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "distance"])
    sigma = edges_df["distance"].replace(0, np.nan).mean()
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    edges_df["weight"] = np.exp(-edges_df["distance"] / sigma)
    return edges_df


def build_cluster_time_matrix(
    agg_clusters: pd.DataFrame,
    fecha_col: str = "fecha",
    cluster_col: str = "cluster",
    value_col: str = "congestion",
    interpolate: bool = True,
) -> pd.DataFrame:
    tabla = (
        agg_clusters.pivot(index=fecha_col, columns=cluster_col, values=value_col)
        .sort_index()
    )
    tabla.index = pd.to_datetime(tabla.index)
    if interpolate:
        tabla = tabla.interpolate(limit_direction="both").fillna(0)
    return tabla


def reindex_time_matrix(tabla_clusters: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Reindexa columnas de clusters a 0..N-1 para evitar errores en torch_geometric."""
    tabla = tabla_clusters.copy()
    nodos = sorted(tabla.columns.tolist())
    mapping = {cluster: i for i, cluster in enumerate(nodos)}
    mapping_inverso = {i: cluster for cluster, i in mapping.items()}
    tabla = tabla[nodos].copy()
    tabla.columns = [mapping[c] for c in tabla.columns]
    tabla = tabla.reindex(sorted(tabla.columns), axis=1)
    return tabla, mapping, mapping_inverso


def adjacency_correlation_topk(
    tabla_clusters: pd.DataFrame,
    k: int = 5,
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], dict[int, int]]:
    tabla = tabla_clusters.copy()

    if drop_constant:
        std_por_cluster = tabla.std()
        clusters_constantes = std_por_cluster[std_por_cluster == 0].index.tolist()
        tabla = tabla.drop(columns=clusters_constantes)

    tabla_gnn, mapping, mapping_inverso = reindex_time_matrix(tabla)
    corr_matrix = tabla_gnn.corr(method="pearson").fillna(0.0)
    corr_values = corr_matrix.values
    cluster_labels = corr_matrix.columns.to_list()

    edges = []
    n = corr_values.shape[0]
    if n == 1:
        edges.append((0, 0, 1.0))
    else:
        k_eff = min(k, n - 1)
        for i in range(n):
            fila = corr_values[i].copy()
            fila[i] = -np.inf
            top_idx = np.argsort(fila)[-k_eff:]
            top_idx = top_idx[np.argsort(fila[top_idx])[::-1]]
            source_cluster = cluster_labels[i]
            for j in top_idx:
                target_cluster = cluster_labels[j]
                weight = float(max(corr_values[i, j], 0.0))
                edges.append((source_cluster, target_cluster, weight))

    edges_gnn = pd.DataFrame(edges, columns=["source", "target", "weight"])
    return tabla_gnn, edges_gnn, mapping, mapping_inverso


def to_pyg_tensors(edges_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    if edges_df.empty:
        raise ValueError("edges_df está vacío; no se puede construir edge_index.")
    edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)
    if "weight" in edges_df.columns:
        weights = edges_df["weight"].values
    else:
        weights = np.ones(len(edges_df), dtype=np.float32)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight
