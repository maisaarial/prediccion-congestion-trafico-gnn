from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString


def _bearing_from_points(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    return (angle + 360.0) % 360.0


def _bearing_to_cardinal(bearing: float) -> str:
    if 45 <= bearing < 135:
        return "E"
    if 135 <= bearing < 225:
        return "S"
    if 225 <= bearing < 315:
        return "W"
    return "N"


def _edge_bearing(G: nx.MultiDiGraph, u, v, key) -> float:
    data = G.get_edge_data(u, v, key)
    geom = data.get("geometry")

    if isinstance(geom, LineString):
        x1, y1 = geom.coords[0]
        x2, y2 = geom.coords[-1]
    else:
        x1 = G.nodes[u]["x"]
        y1 = G.nodes[u]["y"]
        x2 = G.nodes[v]["x"]
        y2 = G.nodes[v]["y"]

    return _bearing_from_points(x1, y1, x2, y2)


def calcular_sentido(
    puntos_medida: pd.DataFrame,
    graphml_path: str | Path,
    id_col: str = "id",
    lat_col: str = "latitud",
    lon_col: str = "longitud",
) -> pd.DataFrame:
    G = ox.load_graphml(graphml_path)
    xs = puntos_medida[lon_col].tolist()
    ys = puntos_medida[lat_col].tolist()
    nearest = ox.distance.nearest_edges(G, X=xs, Y=ys)

    rows = []
    for sensor_id, edge in zip(puntos_medida[id_col], nearest):
        u, v, key = edge
        bearing = _edge_bearing(G, u, v, key)
        rows.append(
            {
                id_col: sensor_id,
                "u": u,
                "v": v,
                "key": key,
                "bearing": bearing,
                "sentido_v1": _bearing_to_cardinal(bearing),
            }
        )

    return pd.DataFrame(rows)


def calcular_sentido_mejorado(
    puntos_medida: pd.DataFrame,
    graphml_path: str | Path,
    id_col: str = "id",
    lat_col: str = "latitud",
    lon_col: str = "longitud",
) -> pd.DataFrame:
    df = calcular_sentido(
        puntos_medida=puntos_medida,
        graphml_path=graphml_path,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
    ).copy()

    df["sentido_v2"] = df["bearing"].apply(lambda b: "NS" if (b < 45 or b >= 315 or 135 <= b < 225) else "EO")
    return df
