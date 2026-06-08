from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class MappingState:
    mode: str
    can_aggregate: bool
    sensor_locations: pd.DataFrame
    sensor_to_node: pd.DataFrame
    nodes_metadata: pd.DataFrame
    node_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_mapping_state(config: dict, project_root: Path) -> MappingState:
    paths_cfg = config.get("paths", {})
    aggregation_cfg = config.get("aggregation", {})
    notes: list[str] = []

    sensors_path = resolve_path(project_root, paths_cfg.get("sensors_csv", ""))
    sensors = _load_sensor_locations(sensors_path, notes)

    graph_meta = _load_graph_metadata(
        graph_pt=resolve_path(project_root, paths_cfg.get("graph_pt", "")),
        tabla_gnn_csv=resolve_path(project_root, paths_cfg.get("tabla_gnn_csv", "")),
        notes=notes,
    )

    if sensors.empty:
        notes.append("No sensor location CSV was available.")
        return MappingState(
            mode="sensor_points",
            can_aggregate=False,
            sensor_locations=sensors,
            sensor_to_node=pd.DataFrame(),
            nodes_metadata=pd.DataFrame(),
            notes=notes,
        )

    try:
        clustered = _cluster_sensors_to_target(sensors, aggregation_cfg)
        sensor_to_node = _build_sensor_to_node(clustered, graph_meta)
    except Exception as exc:
        notes.append(f"Could not rebuild provisional cluster mapping: {exc}")
        sensor_to_node = pd.DataFrame()

    if sensor_to_node.empty:
        node_ids = _node_ids_from_graph_meta(graph_meta)
        notes.append("Falling back to direct XML measurement points.")
        return MappingState(
            mode="sensor_points",
            can_aggregate=False,
            sensor_locations=sensors,
            sensor_to_node=sensor_to_node,
            nodes_metadata=_empty_nodes_metadata(node_ids),
            node_ids=node_ids,
            notes=notes,
        )

    nodes_metadata = _build_nodes_metadata(clustered, sensor_to_node)
    node_ids = nodes_metadata["node_id"].astype(str).tolist()

    return MappingState(
        mode="proximidad_500_correlacion_best_effort",
        can_aggregate=True,
        sensor_locations=sensors,
        sensor_to_node=sensor_to_node,
        nodes_metadata=nodes_metadata,
        node_ids=node_ids,
        notes=notes,
    )


def attach_sensor_metadata(raw_df: pd.DataFrame, mapping_state: MappingState) -> pd.DataFrame:
    out = raw_df.copy()
    out["idelem_int"] = pd.to_numeric(out["idelem"], errors="coerce").astype("Int64")

    if not mapping_state.sensor_locations.empty:
        sensors = mapping_state.sensor_locations[
            ["id", "nombre", "distrito", "utm_x", "utm_y", "longitud", "latitud"]
        ].copy()
        sensors = sensors.rename(
            columns={
                "id": "sensor_id",
                "nombre": "sensor_nombre",
                "longitud": "sensor_lon",
                "latitud": "sensor_lat",
            }
        )
        out = out.merge(
            sensors,
            left_on="idelem_int",
            right_on="sensor_id",
            how="left",
        )
    else:
        out["sensor_id"] = pd.NA
        out["sensor_nombre"] = pd.NA
        out["distrito"] = pd.NA
        out["utm_x"] = pd.NA
        out["utm_y"] = pd.NA
        out["sensor_lon"] = pd.NA
        out["sensor_lat"] = pd.NA

    out["xml_lon"] = pd.NA
    out["xml_lat"] = pd.NA

    st_x = pd.to_numeric(out.get("st_x"), errors="coerce")
    st_y = pd.to_numeric(out.get("st_y"), errors="coerce")

    normal_wgs84 = st_x.between(-10, 10) & st_y.between(35, 45)
    swapped_wgs84 = st_y.between(-10, 10) & st_x.between(35, 45)

    out.loc[normal_wgs84, "xml_lon"] = st_x[normal_wgs84]
    out.loc[normal_wgs84, "xml_lat"] = st_y[normal_wgs84]
    out.loc[swapped_wgs84, "xml_lon"] = st_y[swapped_wgs84]
    out.loc[swapped_wgs84, "xml_lat"] = st_x[swapped_wgs84]

    out["lon"] = pd.to_numeric(out["xml_lon"], errors="coerce").combine_first(
        pd.to_numeric(out["sensor_lon"], errors="coerce")
    )
    out["lat"] = pd.to_numeric(out["xml_lat"], errors="coerce").combine_first(
        pd.to_numeric(out["sensor_lat"], errors="coerce")
    )

    out["descripcion"] = out["descripcion"].fillna(out["sensor_nombre"])
    return out


def _load_sensor_locations(path: Path, notes: list[str]) -> pd.DataFrame:
    if not path.exists():
        notes.append(f"Sensor CSV not found: {path}")
        return pd.DataFrame()

    sensors = pd.read_csv(path, sep=";", encoding="latin-1")
    required = ["id", "utm_x", "utm_y", "longitud", "latitud"]
    missing = [column for column in required if column not in sensors.columns]
    if missing:
        notes.append(f"Sensor CSV is missing columns: {missing}")
        return pd.DataFrame()

    sensors = sensors.copy()
    sensors["id"] = pd.to_numeric(sensors["id"], errors="coerce").astype("Int64")
    for column in ["utm_x", "utm_y", "longitud", "latitud"]:
        sensors[column] = pd.to_numeric(sensors[column], errors="coerce")

    sensors = sensors.dropna(subset=["id", "utm_x", "utm_y"]).copy()
    sensors["id"] = sensors["id"].astype(int)
    return sensors


def _load_graph_metadata(graph_pt: Path, tabla_gnn_csv: Path, notes: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}

    if graph_pt.exists():
        try:
            import torch

            graph = torch.load(graph_pt, map_location="cpu", weights_only=False)
            if isinstance(graph, dict):
                metadata.update(graph)
                notes.append(f"Loaded graph metadata from {graph_pt}")
            else:
                notes.append(f"graph.pt did not contain a dict: {type(graph)!r}")
        except Exception as exc:
            notes.append(f"Could not read graph.pt: {exc}")
    else:
        notes.append(f"graph.pt not found: {graph_pt}")

    if "num_nodes" not in metadata and tabla_gnn_csv.exists():
        try:
            columns = pd.read_csv(tabla_gnn_csv, nrows=0).columns.tolist()
            node_columns = [column for column in columns if column != "fecha"]
            metadata["num_nodes"] = len(node_columns)
            metadata["node_ids_from_table"] = [str(column) for column in node_columns]
            notes.append(f"Read node ids from tabla_gnn.csv: {len(node_columns)} nodes")
        except Exception as exc:
            notes.append(f"Could not read tabla_gnn.csv header: {exc}")

    return metadata


def _cluster_sensors_to_target(sensors: pd.DataFrame, aggregation_cfg: dict) -> pd.DataFrame:
    from sklearn.cluster import DBSCAN

    target = int(aggregation_cfg.get("proximity_target_nodes", 500))
    min_samples = int(aggregation_cfg.get("min_samples", 1))
    eps_min = float(aggregation_cfg.get("proximity_eps_search_min", 10))
    eps_max = float(aggregation_cfg.get("proximity_eps_search_max", 1000))
    iterations = int(aggregation_cfg.get("proximity_eps_search_iter", 25))

    coords = sensors[["utm_x", "utm_y"]].copy()
    best_labels: pd.Series | None = None
    best_diff = float("inf")

    low = eps_min
    high = eps_max
    for _ in range(iterations):
        eps = (low + high) / 2
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = pd.Series(model.fit_predict(coords), index=sensors.index)
        labels = _relabel_with_singletons(labels)
        count = labels.nunique()
        diff = abs(count - target)

        if diff < best_diff:
            best_labels = labels
            best_diff = diff

        if count > target:
            low = eps
        else:
            high = eps

    if best_labels is None:
        raise RuntimeError("DBSCAN did not produce labels")

    clustered = sensors.copy()
    clustered["cluster_proximidad_target"] = best_labels.astype(int)
    clustered["cluster_reindexed"] = _reindex_series(clustered["cluster_proximidad_target"])
    return clustered


def _relabel_with_singletons(labels: pd.Series) -> pd.Series:
    labels = labels.copy()
    next_label = int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
    for idx in labels[labels == -1].index:
        labels.loc[idx] = next_label
        next_label += 1
    return _reindex_series(labels)


def _reindex_series(values: pd.Series) -> pd.Series:
    unique_values = sorted(values.dropna().unique().tolist())
    mapping = {old: new for new, old in enumerate(unique_values)}
    return values.map(mapping).astype(int)


def _build_sensor_to_node(clustered: pd.DataFrame, graph_meta: dict[str, object]) -> pd.DataFrame:
    cluster_to_node = _cluster_to_node_mapping(graph_meta)
    if not cluster_to_node:
        num_nodes = int(graph_meta.get("num_nodes", 0) or 0)
        if num_nodes > 0:
            cluster_to_node = {
                int(cluster): int(cluster)
                for cluster in clustered["cluster_reindexed"].unique()
                if int(cluster) < num_nodes
            }

    if not cluster_to_node:
        return pd.DataFrame()

    out = clustered[["id", "cluster_reindexed"]].copy()
    out["node_id"] = out["cluster_reindexed"].map(cluster_to_node)
    out = out.dropna(subset=["node_id"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["node_id"] = out["node_id"].astype(int).astype(str)
    out = out.rename(columns={"id": "sensor_id", "cluster_reindexed": "cluster_id"})
    return out[["sensor_id", "cluster_id", "node_id"]]


def _cluster_to_node_mapping(graph_meta: dict[str, object]) -> dict[int, int]:
    mapping = graph_meta.get("tabla_columns_mapping")
    if isinstance(mapping, dict) and mapping:
        return {int(key): int(value) for key, value in mapping.items()}

    inverse = graph_meta.get("tabla_columns_mapping_inverso")
    if isinstance(inverse, dict) and inverse:
        return {int(value): int(key) for key, value in inverse.items()}

    return {}


def _build_nodes_metadata(clustered: pd.DataFrame, sensor_to_node: pd.DataFrame) -> pd.DataFrame:
    merged = clustered.merge(sensor_to_node, left_on="id", right_on="sensor_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    nodes = (
        merged.groupby("node_id")
        .agg(
            lat=("latitud", "mean"),
            lon=("longitud", "mean"),
            utm_x=("utm_x", "mean"),
            utm_y=("utm_y", "mean"),
            sensor_count_total=("id", "nunique"),
            sample_sensor_ids=("id", lambda values: ",".join(map(str, sorted(values.unique())[:8]))),
        )
        .reset_index()
    )
    nodes["entity_type"] = "node"
    nodes["node_id_sort"] = pd.to_numeric(nodes["node_id"], errors="coerce")
    return nodes.sort_values("node_id_sort").drop(columns=["node_id_sort"]).reset_index(drop=True)


def _node_ids_from_graph_meta(graph_meta: dict[str, object]) -> list[str]:
    node_ids = graph_meta.get("node_ids_from_table")
    if isinstance(node_ids, list) and node_ids:
        return [str(node_id) for node_id in node_ids]

    num_nodes = int(graph_meta.get("num_nodes", 0) or 0)
    return [str(node_id) for node_id in range(num_nodes)]


def _empty_nodes_metadata(node_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"node_id": node_ids, "entity_type": "node_artifact"})
