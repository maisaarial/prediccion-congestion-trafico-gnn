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
    coords = centroides[[x_col, y_col]].values
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(coords)
    distancias, indices = nbrs.kneighbors(coords)

    edges = []
    for i in range(len(coords)):
        for dist, j in zip(distancias[i, 1:], indices[i, 1:]):
            edges.append((i, j, dist))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "distance"])
    sigma = edges_df["distance"].mean() if not edges_df.empty else 1.0
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

    corr_matrix = tabla.corr(method="pearson")
    corr_values = corr_matrix.values
    cluster_labels = corr_matrix.columns.to_list()

    edges = []
    n = corr_values.shape[0]
    for i in range(n):
        fila = corr_values[i].copy()
        fila[i] = -np.inf
        top_idx = np.argsort(fila)[-k:]
        top_idx = top_idx[np.argsort(fila[top_idx])[::-1]]
        source_cluster = cluster_labels[i]
        for j in top_idx:
            target_cluster = cluster_labels[j]
            edges.append((source_cluster, target_cluster, corr_values[i, j]))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    edges_df = edges_df.sort_values(["source", "weight"], ascending=[True, False]).reset_index(drop=True)

    nodos_validos = sorted(tabla.columns.tolist())
    mapping = {cluster: i for i, cluster in enumerate(nodos_validos)}
    mapping_inverso = {i: cluster for cluster, i in mapping.items()}

    tabla_gnn = tabla[nodos_validos].copy()
    tabla_gnn.columns = [mapping[c] for c in tabla_gnn.columns]
    tabla_gnn = tabla_gnn.reindex(sorted(tabla_gnn.columns), axis=1)

    edges_gnn = edges_df.copy()
    edges_gnn["source"] = edges_gnn["source"].map(mapping)
    edges_gnn["target"] = edges_gnn["target"].map(mapping)

    return tabla_gnn, edges_gnn, mapping, mapping_inverso



def to_pyg_tensors(edges_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)
    edge_weight = torch.tensor(edges_df["weight"].values, dtype=torch.float32)
    return edge_index, edge_weight
