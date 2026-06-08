from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd


REAL_MODEL_MODE = "REAL_MODEL_TEMPORAL"
HISTORY_INSUFFICIENT_MODE = "DEMO_HISTORICO_INSUFICIENTE"
ARTIFACTS_MISSING_MODE = "DEMO_ARTEFACTOS_NO_DISPONIBLES"
FALLBACK_ERROR_MODE = "DEMO_FALLBACK_ERROR"

REQUIRED_ARTIFACTS = ["model.pt", "graph.pt", "edges.csv", "tabla_gnn.csv", "model_config.json"]
DEFAULT_WARNING = "Modelo temporal de integracion. No corresponde al modelo final de calidad predictiva."


@dataclass
class WindowBundle:
    frame: pd.DataFrame
    window_available: int
    window_required: int
    block_ids: list[str]


@dataclass
class PredictionBundle:
    predictions: pd.DataFrame | None
    status: dict[str, Any]


class RealModelRuntime:
    def __init__(self, project_root: Path, model_dir: Path):
        self.project_root = project_root
        self.model_dir = model_dir
        self.config: dict[str, Any] = {}
        self.graph: dict[str, Any] = {}
        self.edges_df = pd.DataFrame()
        self.model: Any = None
        self.edge_index: Any = None
        self.edge_weight: Any = None
        self.node_order: list[str] = []
        self.device = "cpu"
        self.loaded = False
        self.load_error: str | None = None

    @property
    def model_available(self) -> bool:
        return all((self.model_dir / filename).exists() for filename in REQUIRED_ARTIFACTS)

    @property
    def missing_artifacts(self) -> list[str]:
        return [filename for filename in REQUIRED_ARTIFACTS if not (self.model_dir / filename).exists()]

    def load(self) -> bool:
        if self.loaded:
            return True
        if self.load_error is not None:
            return False
        if not self.model_available:
            self.load_error = f"Faltan artefactos: {', '.join(self.missing_artifacts)}"
            return False

        try:
            import torch

            src_path = self.project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from traffic_gnn.models.gcn_gru import GCN_GRU

            self.config = _load_json(self.model_dir / "model_config.json")
            if self.config.get("model") != "GCN_GRU":
                raise ValueError(f"Modelo no soportado en MVP real-time: {self.config.get('model')}")

            self.graph = torch.load(self.model_dir / "graph.pt", map_location="cpu", weights_only=False)
            if not isinstance(self.graph, dict):
                raise TypeError(f"graph.pt debe contener dict, recibido: {type(self.graph)!r}")
            self.edges_df = pd.read_csv(self.model_dir / "edges.csv")
            if self.edges_df.empty:
                raise ValueError("edges.csv esta vacio.")

            _validate_graph_config(self.config, self.graph)

            num_nodes = int(self.config.get("num_nodes") or self.graph["num_nodes"])
            features = int(self.config.get("features") or self.graph.get("num_features", 1))
            hidden_channels = int(self.config.get("hidden_channels", 32))
            recurrent_hidden = int(self.config.get("recurrent_hidden", 64))

            self.model = GCN_GRU(
                num_nodes=num_nodes,
                in_channels=features,
                hidden_channels=hidden_channels,
                lstm_hidden=recurrent_hidden,
            )
            state_dict = torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.edge_index = self.graph["edge_index"].to(self.device)
            edge_weight = self.graph.get("edge_weight")
            self.edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
            self.node_order = _load_node_order(self.model_dir / "tabla_gnn.csv", num_nodes)

            if len(self.node_order) != num_nodes:
                raise ValueError(
                    f"tabla_gnn.csv define {len(self.node_order)} nodos, pero config espera {num_nodes}"
                )

            self.loaded = True
            return True
        except Exception as exc:
            self.load_error = str(exc)
            return False

    def status(
        self,
        blocks_dir: Path | None = None,
        mode: str | None = None,
        window_available: int | None = None,
    ) -> dict[str, Any]:
        if self.model_available and not self.loaded and self.load_error is None:
            self.load()

        config = self.config if self.config else _try_load_json(self.model_dir / "model_config.json")
        window_required = int(config.get("window", 12) or 12)
        if window_available is None and blocks_dir is not None:
            window_available = count_available_windows(blocks_dir)
        if window_available is None:
            window_available = 0

        if mode is None:
            if not self.model_available:
                mode = ARTIFACTS_MISSING_MODE
            elif window_available < window_required:
                mode = HISTORY_INSUFFICIENT_MODE
            elif self.loaded:
                mode = REAL_MODEL_MODE
            else:
                mode = FALLBACK_ERROR_MODE

        warning = config.get("warning") or DEFAULT_WARNING
        if self.load_error and mode == FALLBACK_ERROR_MODE:
            warning = f"{warning} Error de carga/inferencia: {self.load_error}"

        return {
            "model_available": self.model_available,
            "model_loaded": self.loaded,
            "mode": mode,
            "model": config.get("model"),
            "case": config.get("case"),
            "adjacency": config.get("adjacency"),
            "num_nodes": int(config.get("num_nodes", 0) or 0),
            "window_required": window_required,
            "window_available": int(window_available),
            "warning": warning,
            "missing_artifacts": self.missing_artifacts,
            "load_error": self.load_error,
        }


_RUNTIME_CACHE: dict[Path, RealModelRuntime] = {}


def get_runtime(project_root: Path, model_dir: Path) -> RealModelRuntime:
    model_dir = model_dir.resolve()
    runtime = _RUNTIME_CACHE.get(model_dir)
    if runtime is None:
        runtime = RealModelRuntime(project_root=project_root, model_dir=model_dir)
        _RUNTIME_CACHE[model_dir] = runtime
    return runtime


def predict_real_temporal(
    current_entities: pd.DataFrame,
    project_root: Path,
    model_dir: Path,
    blocks_dir: Path,
) -> PredictionBundle:
    runtime = get_runtime(project_root, model_dir)
    config = _try_load_json(model_dir / "model_config.json")
    window_required = int(config.get("window", 12) or 12)
    node_order = _load_node_order(model_dir / "tabla_gnn.csv", int(config.get("num_nodes", 0) or 0))
    window = build_recent_window(blocks_dir, window_required=window_required, node_order=node_order)

    if not runtime.model_available:
        return PredictionBundle(
            predictions=None,
            status=runtime.status(blocks_dir, mode=ARTIFACTS_MISSING_MODE, window_available=window.window_available),
        )

    if window.window_available < window_required:
        return PredictionBundle(
            predictions=None,
            status=runtime.status(blocks_dir, mode=HISTORY_INSUFFICIENT_MODE, window_available=window.window_available),
        )

    try:
        predicted_by_node = predict_with_real_model(window.frame, project_root=project_root, model_dir=model_dir)
        predictions = current_entities.copy()
        predictions["node_id"] = predictions["node_id"].astype(str)
        predictions["prediccion_15min"] = predictions["node_id"].map(predicted_by_node)
        predictions["prediccion_15min"] = pd.to_numeric(
            predictions["prediccion_15min"], errors="coerce"
        ).clip(0.0, 1.0)
        current = pd.to_numeric(predictions["congestion_actual"], errors="coerce")
        predictions["diferencia"] = predictions["prediccion_15min"] - current
        predictions["predictor"] = "real_gcn_gru_temporal"
        predictions["modo_modelo"] = REAL_MODEL_MODE
        return PredictionBundle(
            predictions=predictions,
            status=runtime.status(blocks_dir, mode=REAL_MODEL_MODE, window_available=window.window_available),
        )
    except Exception as exc:
        runtime.load_error = str(exc)
        return PredictionBundle(
            predictions=None,
            status=runtime.status(blocks_dir, mode=FALLBACK_ERROR_MODE, window_available=window.window_available),
        )


def predict_with_real_model(
    window_df: pd.DataFrame,
    project_root: Path | None = None,
    model_dir: Path | None = None,
) -> dict[str, float]:
    project_root = project_root or Path(__file__).resolve().parents[3]
    model_dir = model_dir or project_root / "realtime_app" / "artifacts" / "selected_model"
    runtime = get_runtime(project_root, model_dir)
    if not runtime.load():
        raise RuntimeError(runtime.load_error or "No se pudo cargar el modelo real temporal.")

    import torch

    x_values = _window_to_numpy(window_df, runtime.node_order)
    x_seq = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        y_pred = runtime.model(x_seq, runtime.edge_index, runtime.edge_weight)

    values = y_pred.squeeze(0).detach().cpu().numpy()
    return {node_id: float(value) for node_id, value in zip(runtime.node_order, values)}


def get_model_status(project_root: Path, model_dir: Path, blocks_dir: Path) -> dict[str, Any]:
    runtime = get_runtime(project_root, model_dir)
    return runtime.status(blocks_dir=blocks_dir)


def build_recent_window(blocks_dir: Path, window_required: int, node_order: list[str]) -> WindowBundle:
    snapshots = _load_block_snapshots(blocks_dir)
    window_available = len(snapshots)
    if window_available < window_required:
        return WindowBundle(
            frame=pd.DataFrame(),
            window_available=window_available,
            window_required=window_required,
            block_ids=[snapshot["block_id"] for snapshot in snapshots],
        )

    selected = snapshots[-window_required:]
    frames = []
    for index, snapshot in enumerate(selected):
        frame = snapshot["frame"].copy()
        frame["__window_index"] = index
        frame["__block_id"] = snapshot["block_id"]
        frames.append(frame)

    return WindowBundle(
        frame=pd.concat(frames, ignore_index=True),
        window_available=window_available,
        window_required=window_required,
        block_ids=[snapshot["block_id"] for snapshot in selected],
    )


def count_available_windows(blocks_dir: Path) -> int:
    return len(_load_block_snapshots(blocks_dir))


def _window_to_numpy(window_df: pd.DataFrame, node_order: list[str]) -> Any:
    import numpy as np

    if window_df.empty:
        raise ValueError("window_df esta vacio.")
    if "__window_index" not in window_df.columns:
        window_df = _assign_window_index(window_df)

    values = []
    for _, block_df in window_df.sort_values("__window_index").groupby("__window_index", sort=True):
        node_values = (
            block_df.assign(node_id=block_df["node_id"].astype(str))
            .set_index("node_id")["congestion_actual"]
            .pipe(pd.to_numeric, errors="coerce")
            .reindex(node_order)
            .fillna(0.0)
            .clip(0.0, 1.0)
            .to_numpy(dtype="float32")
        )
        values.append(node_values)

    return np.stack(values, axis=0)


def _assign_window_index(window_df: pd.DataFrame) -> pd.DataFrame:
    out = window_df.copy()
    block_col = "bloque_15min" if "bloque_15min" in out.columns else "fecha_hora"
    block_values = out[block_col].astype(str).fillna("")
    order = {value: index for index, value in enumerate(dict.fromkeys(block_values.tolist()))}
    out["__window_index"] = block_values.map(order)
    return out


def _load_block_snapshots(blocks_dir: Path) -> list[dict[str, Any]]:
    if not blocks_dir.exists():
        return []

    latest_by_block: dict[str, dict[str, Any]] = {}
    for path in sorted(blocks_dir.glob("nodes_15min_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "node_id" not in frame.columns or "congestion_actual" not in frame.columns:
            continue

        block_id = _block_id(frame, path)
        capture_order = _capture_order(path)
        snapshot = {
            "path": path,
            "frame": frame,
            "block_id": block_id,
            "capture_order": capture_order,
        }
        previous = latest_by_block.get(block_id)
        if previous is None or capture_order >= previous["capture_order"]:
            latest_by_block[block_id] = snapshot

    return sorted(latest_by_block.values(), key=lambda item: item["capture_order"])


def _block_id(frame: pd.DataFrame, path: Path) -> str:
    if "bloque_15min" in frame.columns:
        values = frame["bloque_15min"].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return path.stem


def _capture_order(path: Path) -> datetime:
    match = re.search(r"(\d{8}_\d{6})$", path.stem)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _load_node_order(tabla_gnn_csv: Path, num_nodes: int) -> list[str]:
    if tabla_gnn_csv.exists():
        columns = pd.read_csv(tabla_gnn_csv, nrows=0).columns.tolist()
        node_columns = [column for column in columns if column != "fecha"]
        if node_columns:
            return [str(column) for column in node_columns]
    return [str(node_id) for node_id in range(num_nodes)]


def _validate_graph_config(config: dict[str, Any], graph: dict[str, Any]) -> None:
    expected_pairs = [
        ("case", "caso"),
        ("adjacency", "tipo_adyacencia"),
    ]
    for config_key, graph_key in expected_pairs:
        config_value = config.get(config_key)
        graph_value = graph.get(graph_key)
        if config_value and graph_value and str(config_value) != str(graph_value):
            raise ValueError(
                f"Artefactos incompatibles: config {config_key}={config_value}, "
                f"graph {graph_key}={graph_value}"
            )

    config_nodes = config.get("num_nodes")
    graph_nodes = graph.get("num_nodes")
    if config_nodes is not None and graph_nodes is not None and int(config_nodes) != int(graph_nodes):
        raise ValueError(
            f"Artefactos incompatibles: config num_nodes={config_nodes}, graph num_nodes={graph_nodes}"
        )

    config_features = config.get("features")
    graph_features = graph.get("num_features")
    if config_features is not None and graph_features is not None and int(config_features) != int(graph_features):
        raise ValueError(
            f"Artefactos incompatibles: config features={config_features}, graph num_features={graph_features}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _try_load_json(path: Path) -> dict[str, Any]:
    try:
        return _load_json(path)
    except Exception:
        return {}
