from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .node_mapping import MappingState


def add_demo_congestion(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()
    ocupacion = pd.to_numeric(out["ocupacion"], errors="coerce")
    # Demo signal only: each sensor uses ocupacion / 100 clipped to the 0..1 range.
    out["congestion_demo"] = (ocupacion / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    return out


def add_time_block(raw_df: pd.DataFrame, frequency_minutes: int = 15) -> pd.DataFrame:
    out = raw_df.copy()
    timestamps = pd.to_datetime(out["fecha_hora"], errors="coerce")
    out["bloque_15min"] = timestamps.dt.floor(f"{frequency_minutes}min")
    return out


def build_current_entities(
    raw_df: pd.DataFrame,
    mapping_state: MappingState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapping_state.can_aggregate and not mapping_state.sensor_to_node.empty:
        mapped_raw = raw_df.merge(
            mapping_state.sensor_to_node,
            left_on="idelem_int",
            right_on="sensor_id",
            how="left",
        )
        mapped_ok = mapped_raw.dropna(subset=["node_id"]).copy()
        if not mapped_ok.empty:
            nodes = _aggregate_nodes(mapped_ok, mapping_state)
            if not nodes.empty:
                return nodes, mapped_raw

    sensor_points = _build_sensor_points(raw_df, mapping_state)
    return sensor_points, raw_df.copy()


def save_dataframe_snapshot(
    df: pd.DataFrame,
    directory: Path,
    prefix: str,
    captured_at: datetime,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = captured_at.astimezone().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{prefix}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def _aggregate_nodes(mapped_raw: pd.DataFrame, mapping_state: MappingState) -> pd.DataFrame:
    # Node-level demo congestion is the mean congestion_demo of active sensors in that node.
    nodes = (
        mapped_raw.groupby("node_id")
        .agg(
            congestion_actual=("congestion_demo", "mean"),
            intensidad=("intensidad", "mean"),
            ocupacion=("ocupacion", "mean"),
            carga=("carga", "mean"),
            nivelServicio=("nivelServicio", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            fecha_hora=("fecha_hora", "max"),
            bloque_15min=("bloque_15min", "max"),
            sensor_count=("idelem_int", "nunique"),
        )
        .reset_index()
    )

    meta = mapping_state.nodes_metadata.copy()
    if not meta.empty:
        nodes = nodes.merge(
            meta.drop(columns=["entity_type"], errors="ignore"),
            on="node_id",
            how="left",
            suffixes=("", "_meta"),
        )
        nodes["lat"] = nodes["lat"].combine_first(nodes.get("lat_meta"))
        nodes["lon"] = nodes["lon"].combine_first(nodes.get("lon_meta"))

    nodes["entity_type"] = "node"
    nodes["mapping_mode"] = mapping_state.mode
    nodes["descripcion"] = "Nodo GNN " + nodes["node_id"].astype(str)

    nodes = nodes.drop(columns=[column for column in ["lat_meta", "lon_meta"] if column in nodes.columns])
    nodes["node_id_sort"] = pd.to_numeric(nodes["node_id"], errors="coerce")
    return nodes.sort_values("node_id_sort").drop(columns=["node_id_sort"]).reset_index(drop=True)


def _build_sensor_points(raw_df: pd.DataFrame, mapping_state: MappingState) -> pd.DataFrame:
    points = raw_df.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return pd.DataFrame()

    points["node_id"] = points["idelem_int"].astype(str)
    points["entity_type"] = "sensor"
    points["mapping_mode"] = mapping_state.mode
    points["congestion_actual"] = pd.to_numeric(points["congestion_demo"], errors="coerce").fillna(0.0)
    points["sensor_count"] = 1

    columns = [
        "node_id",
        "entity_type",
        "mapping_mode",
        "descripcion",
        "congestion_actual",
        "intensidad",
        "ocupacion",
        "carga",
        "nivelServicio",
        "lat",
        "lon",
        "fecha_hora",
        "bloque_15min",
        "sensor_count",
    ]
    return points[columns].sort_values("node_id").reset_index(drop=True)
