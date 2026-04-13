from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN



def _relabel_with_singletons(labels: pd.Series) -> pd.Series:
    """Convierte el ruido (-1) en clusters singleton y reindexa consecutivamente."""
    labels = labels.copy()
    next_label = labels[labels >= 0].max() + 1 if (labels >= 0).any() else 0
    for idx in labels[labels == -1].index:
        labels.loc[idx] = next_label
        next_label += 1

    unique_labels = sorted(labels.unique())
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return labels.map(mapping)


def generar_cluster_proximidad(
    sensores: pd.DataFrame,
    epsilon: float,
    min_samples: int = 1,
    x_col: str = "utm_x",
    y_col: str = "utm_y",
    cluster_col: str = "cluster_proximidad",
) -> pd.DataFrame:
    df = sensores.copy()
    coords = df[[x_col, y_col]].copy()

    model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="euclidean")
    labels = pd.Series(model.fit_predict(coords), index=df.index)
    df[cluster_col] = _relabel_with_singletons(labels)
    return df
