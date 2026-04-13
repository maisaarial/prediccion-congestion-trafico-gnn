from __future__ import annotations

"""Plantilla simple para construir adyacencias desde scripts."""

from traffic_gnn.graph.aggregation import aggregation_congestion_por_clusters, calcular_centroides_clusters
from traffic_gnn.graph.adjacency import adjacency_knn_from_centroids, adjacency_correlation_topk, to_pyg_tensors
from traffic_gnn.graph.datasets import crear_ventanas, split_temporal


__all__ = [
    "aggregation_congestion_por_clusters",
    "calcular_centroides_clusters",
    "adjacency_knn_from_centroids",
    "adjacency_correlation_topk",
    "to_pyg_tensors",
    "crear_ventanas",
    "split_temporal",
]
