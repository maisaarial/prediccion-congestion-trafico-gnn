from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

try:
    from realtime_app.backend.services.aggregator import (
        add_demo_congestion,
        add_time_block,
        build_current_entities,
        save_dataframe_snapshot,
    )
    from realtime_app.backend.services.mock_predictor import PREDICTOR_MODE, predict_next_15min
    from realtime_app.backend.services.node_mapping import attach_sensor_metadata, load_mapping_state, resolve_path
    from realtime_app.backend.services.real_predictor import (
        FALLBACK_ERROR_MODE,
        HISTORY_INSUFFICIENT_MODE,
        get_model_status,
        predict_real_temporal,
    )
    from realtime_app.backend.services.xml_fetcher import fetch_xml, load_latest_xml_capture, store_xml_capture
    from realtime_app.backend.services.xml_parser import parse_pm_xml
except ModuleNotFoundError:
    from services.aggregator import (
        add_demo_congestion,
        add_time_block,
        build_current_entities,
        save_dataframe_snapshot,
    )
    from services.mock_predictor import PREDICTOR_MODE, predict_next_15min
    from services.node_mapping import attach_sensor_metadata, load_mapping_state, resolve_path
    from services.real_predictor import (
        FALLBACK_ERROR_MODE,
        HISTORY_INSUFFICIENT_MODE,
        get_model_status,
        predict_real_temporal,
    )
    from services.xml_fetcher import fetch_xml, load_latest_xml_capture, store_xml_capture
    from services.xml_parser import parse_pm_xml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "realtime_app"
CONFIG_PATH = APP_ROOT / "configs" / "realtime.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


CONFIG = load_config()
MAPPING_STATE = load_mapping_state(CONFIG, PROJECT_ROOT)
LOGGER = logging.getLogger("realtime_app.history")
if not LOGGER.handlers:
    log_path = APP_ROOT / "data" / "history_collector.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(file_handler)
LOGGER.setLevel(logging.INFO)

app = FastAPI(
    title="Madrid Traffic GNN Realtime Demo",
    version="0.1.0",
    description="Realtime MVP using live Madrid XML, temporary GNN inference, and mock fallback.",
)

_cache_lock = Lock()
_cache: dict[str, Any] = {
    "updated_at": None,
    "raw": pd.DataFrame(),
    "mapped_raw": pd.DataFrame(),
    "nodes": pd.DataFrame(),
    "predictions": pd.DataFrame(),
    "capture_path": None,
    "parsed_path": None,
    "blocks_path": None,
    "predictions_path": None,
    "from_cache": False,
    "last_error": None,
    "model_status": {},
}
_history_lock = Lock()
_seen_capture_timestamps: set[str] = set()
_history_state: dict[str, Any] = {
    "auto_collector_enabled": True,
    "capture_interval_minutes": 5,
    "block_interval_minutes": 15,
    "required_blocks": 12,
    "available_blocks": 0,
    "window_ready": False,
    "last_capture_timestamp": None,
    "last_block_timestamp": None,
    "last_run_at": None,
    "last_event": None,
    "last_error": None,
}
_collector_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup_history_collector() -> None:
    _initialize_history_state()
    if not _collector_enabled():
        LOGGER.info("Recolector automatico desactivado por configuracion.")
        return

    global _collector_task
    if _collector_task is None or _collector_task.done():
        _collector_task = asyncio.create_task(_history_collector_loop())
        LOGGER.info("Recolector automatico iniciado.")


@app.on_event("shutdown")
async def shutdown_history_collector() -> None:
    global _collector_task
    if _collector_task is None:
        return
    _collector_task.cancel()
    with suppress(asyncio.CancelledError):
        await _collector_task
    _collector_task = None
    LOGGER.info("Recolector automatico detenido.")


@app.get("/health")
def health() -> dict[str, Any]:
    model_status = _current_model_status()
    return {
        "status": "ok",
        "mode": CONFIG.get("mode", "demo"),
        "predictor_mode": model_status.get("mode", PREDICTOR_MODE),
        "model_loaded": bool(model_status.get("model_loaded")),
        "mapping_mode": MAPPING_STATE.mode,
        "can_aggregate_by_node": MAPPING_STATE.can_aggregate,
        "known_node_count": len(MAPPING_STATE.node_ids),
        "last_update": _iso(_cache.get("updated_at")),
        "last_error": _cache.get("last_error"),
        "notes": MAPPING_STATE.notes,
    }


@app.get("/latest_raw")
def latest_raw() -> dict[str, Any]:
    snapshot = _get_latest_snapshot()
    return _payload(snapshot, "raw")


@app.get("/nodes")
def nodes() -> dict[str, Any]:
    snapshot = _get_latest_snapshot()
    return _payload(snapshot, "nodes")


@app.get("/predictions")
def predictions() -> dict[str, Any]:
    snapshot = _get_latest_snapshot()
    return _payload(snapshot, "predictions")


@app.get("/model/status")
def model_status() -> dict[str, Any]:
    return _cache.get("model_status") or _current_model_status()


@app.get("/history/status")
def history_status() -> dict[str, Any]:
    return _current_history_status()


@app.get("/nodes/{node_id}")
def node_detail(node_id: str) -> dict[str, Any]:
    snapshot = _get_latest_snapshot()
    predictions_df = snapshot["predictions"].copy()
    matched = predictions_df[predictions_df["node_id"].astype(str) == str(node_id)]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Node or sensor not found: {node_id}")

    mapped_raw = snapshot["mapped_raw"].copy()
    if "node_id" in mapped_raw.columns:
        observations = mapped_raw[mapped_raw["node_id"].astype(str) == str(node_id)]
    else:
        observations = mapped_raw[mapped_raw["idelem_int"].astype(str) == str(node_id)]

    return {
        "metadata": _metadata(snapshot),
        "node": _records(matched)[0],
        "observations": _records(observations),
    }


def _get_latest_snapshot(force: bool = False) -> dict[str, Any]:
    ttl_seconds = int(CONFIG.get("cache", {}).get("ttl_seconds", 60))
    now = datetime.now(timezone.utc)

    with _cache_lock:
        updated_at = _cache.get("updated_at")
        is_fresh = updated_at is not None and (now - updated_at).total_seconds() < ttl_seconds
        if is_fresh and not force:
            return dict(_cache)

        try:
            _refresh_cache()
        except Exception as exc:
            _cache["last_error"] = str(exc)
            if _cache.get("updated_at") is not None:
                return dict(_cache)
            raise HTTPException(status_code=503, detail=f"Could not build realtime snapshot: {exc}") from exc

        return dict(_cache)


def _refresh_cache(source: str = "api") -> None:
    paths_cfg = CONFIG.get("paths", {})
    source_cfg = CONFIG.get("source", {})
    cache_cfg = CONFIG.get("cache", {})
    aggregation_cfg = CONFIG.get("aggregation", {})

    xml_dir = resolve_path(PROJECT_ROOT, paths_cfg["xml_captures_dir"])
    parsed_dir = resolve_path(PROJECT_ROOT, paths_cfg["parsed_captures_dir"])
    blocks_dir = resolve_path(PROJECT_ROOT, paths_cfg["blocks_15min_dir"])
    predictions_dir = resolve_path(PROJECT_ROOT, paths_cfg["predictions_dir"])

    url = source_cfg["xml_url"]
    parsed_path = _cache.get("parsed_path")
    blocks_path = _cache.get("blocks_path")
    predictions_path = _cache.get("predictions_path")
    capture_path = _cache.get("capture_path")
    from_cache = False
    duplicate_capture = False

    try:
        download = fetch_xml(
            url=url,
            timeout_seconds=int(source_cfg.get("timeout_seconds", 20)),
            user_agent=source_cfg.get("user_agent", "traffic-gnn-realtime-demo/0.1"),
        )
        parsed = parse_pm_xml(download.content, captured_at=download.captured_at)
        capture_timestamp = _capture_timestamp_key(parsed, download.captured_at)
        duplicate_capture = _is_duplicate_capture(capture_timestamp)

        if duplicate_capture:
            LOGGER.info("Captura duplicada ignorada: fecha_hora=%s source=%s", capture_timestamp, source)
            captured_at = download.captured_at
        else:
            capture = store_xml_capture(download, xml_dir)
            capture_path = str(capture.path)
            captured_at = capture.captured_at
            _remember_capture_timestamp(capture_timestamp)
            LOGGER.info("Nueva captura guardada: %s fecha_hora=%s", capture.path, capture_timestamp)
    except Exception:
        if not bool(cache_cfg.get("allow_latest_capture_fallback", True)):
            raise
        capture = load_latest_xml_capture(xml_dir, source_url=url)
        if capture is None:
            raise
        parsed = parse_pm_xml(capture.content, captured_at=capture.captured_at)
        capture_timestamp = _capture_timestamp_key(parsed, capture.captured_at)
        captured_at = capture.captured_at
        capture_path = str(capture.path)
        from_cache = True
        duplicate_capture = True
        LOGGER.info("Usando ultima captura local por fallo de descarga: %s", capture.path)

    raw = add_demo_congestion(parsed)
    raw = attach_sensor_metadata(raw, MAPPING_STATE)
    raw = add_time_block(raw, frequency_minutes=int(aggregation_cfg.get("frequency_minutes", 15)))

    nodes_df, mapped_raw = build_current_entities(raw, MAPPING_STATE)
    block_timestamp = _block_timestamp_key(nodes_df)

    if not duplicate_capture and not from_cache:
        parsed_path = str(save_dataframe_snapshot(raw, parsed_dir, "pm_parsed", captured_at))
        blocks_path = str(save_dataframe_snapshot(nodes_df, blocks_dir, "nodes_15min", captured_at))
        LOGGER.info("Bloque de 15 min creado o actualizado: %s", block_timestamp)

    predictions_df, model_status = _predict_with_real_or_mock(nodes_df, blocks_dir)

    if not duplicate_capture and not from_cache:
        predictions_path = str(save_dataframe_snapshot(predictions_df, predictions_dir, "predictions", captured_at))

    _update_history_after_capture(
        capture_timestamp=capture_timestamp,
        block_timestamp=block_timestamp,
        model_status=model_status,
        duplicate_capture=duplicate_capture,
        source=source,
    )

    _cache.update(
        {
            "updated_at": datetime.now(timezone.utc),
            "raw": raw,
            "mapped_raw": mapped_raw,
            "nodes": nodes_df,
            "predictions": predictions_df,
            "capture_path": capture_path,
            "parsed_path": parsed_path,
            "blocks_path": blocks_path,
            "predictions_path": predictions_path,
            "from_cache": from_cache,
            "last_error": None,
            "model_status": model_status,
        }
    )


async def _history_collector_loop() -> None:
    while True:
        await asyncio.to_thread(_run_history_collection_once, "auto")
        await asyncio.sleep(_capture_interval_minutes() * 60)


def _run_history_collection_once(source: str) -> None:
    try:
        with _cache_lock:
            _refresh_cache(source=source)
    except Exception as exc:
        LOGGER.exception("Error en recolector automatico: %s", exc)
        with _history_lock:
            _history_state["last_error"] = str(exc)
            _history_state["last_run_at"] = datetime.now(timezone.utc).isoformat()
            _history_state["last_event"] = "error"


def _initialize_history_state() -> None:
    paths_cfg = CONFIG.get("paths", {})
    parsed_dir = resolve_path(PROJECT_ROOT, paths_cfg.get("parsed_captures_dir", "realtime_app/data/captures/parsed"))
    blocks_dir = resolve_path(PROJECT_ROOT, paths_cfg.get("blocks_15min_dir", "realtime_app/data/blocks_15min"))
    existing_timestamps = _scan_existing_capture_timestamps(parsed_dir)
    latest_capture = _latest_capture_timestamp_from_files(parsed_dir)
    latest_block = _latest_block_timestamp(blocks_dir)
    model_status = _current_model_status()

    with _history_lock:
        _seen_capture_timestamps.update(existing_timestamps)
        _history_state.update(
            {
                "auto_collector_enabled": _collector_enabled(),
                "capture_interval_minutes": _capture_interval_minutes(),
                "block_interval_minutes": _block_interval_minutes(),
                "required_blocks": int(model_status.get("window_required", 12) or 12),
                "available_blocks": int(model_status.get("window_available", 0) or 0),
                "window_ready": bool(
                    int(model_status.get("window_available", 0) or 0)
                    >= int(model_status.get("window_required", 12) or 12)
                ),
                "last_capture_timestamp": latest_capture or _latest_timestamp(existing_timestamps),
                "last_block_timestamp": latest_block,
            }
        )


def _current_history_status() -> dict[str, Any]:
    _initialize_history_state()
    model_status = _current_model_status()
    available_blocks = int(model_status.get("window_available", 0) or 0)
    required_blocks = int(model_status.get("window_required", 12) or 12)

    with _history_lock:
        _history_state.update(
            {
                "auto_collector_enabled": _collector_enabled(),
                "capture_interval_minutes": _capture_interval_minutes(),
                "block_interval_minutes": _block_interval_minutes(),
                "required_blocks": required_blocks,
                "available_blocks": available_blocks,
                "window_ready": available_blocks >= required_blocks,
                "last_block_timestamp": _latest_block_timestamp(
                    resolve_path(
                        PROJECT_ROOT,
                        CONFIG.get("paths", {}).get("blocks_15min_dir", "realtime_app/data/blocks_15min"),
                    )
                ),
            }
        )
        return {
            "auto_collector_enabled": bool(_history_state["auto_collector_enabled"]),
            "capture_interval_minutes": int(_history_state["capture_interval_minutes"]),
            "block_interval_minutes": int(_history_state["block_interval_minutes"]),
            "required_blocks": int(_history_state["required_blocks"]),
            "available_blocks": int(_history_state["available_blocks"]),
            "window_ready": bool(_history_state["window_ready"]),
            "last_capture_timestamp": _history_state["last_capture_timestamp"],
            "last_block_timestamp": _history_state["last_block_timestamp"],
            "last_run_at": _history_state["last_run_at"],
            "last_event": _history_state["last_event"],
            "last_error": _history_state["last_error"],
        }


def _update_history_after_capture(
    capture_timestamp: str,
    block_timestamp: str | None,
    model_status: dict[str, Any],
    duplicate_capture: bool,
    source: str,
) -> None:
    available_blocks = int(model_status.get("window_available", 0) or 0)
    required_blocks = int(model_status.get("window_required", 12) or 12)
    window_ready = available_blocks >= required_blocks
    event = "duplicate_ignored" if duplicate_capture else "capture_saved"

    with _history_lock:
        _history_state.update(
            {
                "auto_collector_enabled": _collector_enabled(),
                "capture_interval_minutes": _capture_interval_minutes(),
                "block_interval_minutes": _block_interval_minutes(),
                "required_blocks": required_blocks,
                "available_blocks": available_blocks,
                "window_ready": window_ready,
                "last_capture_timestamp": capture_timestamp,
                "last_block_timestamp": block_timestamp or _history_state.get("last_block_timestamp"),
                "last_run_at": datetime.now(timezone.utc).isoformat(),
                "last_event": event,
                "last_error": None,
            }
        )

    LOGGER.info(
        "Bloques disponibles: %s/%s. Ventana de 12 bloques lista: %s. source=%s",
        available_blocks,
        required_blocks,
        "si" if window_ready else "no",
        source,
    )


def _collector_enabled() -> bool:
    return bool(CONFIG.get("collector", {}).get("enabled", True))


def _capture_interval_minutes() -> int:
    return int(CONFIG.get("collector", {}).get("capture_interval_minutes", 5) or 5)


def _block_interval_minutes() -> int:
    return int(CONFIG.get("aggregation", {}).get("frequency_minutes", 15) or 15)


def _capture_timestamp_key(parsed: pd.DataFrame, fallback: datetime) -> str:
    if not parsed.empty and "fecha_hora" in parsed.columns:
        timestamps = pd.to_datetime(parsed["fecha_hora"], errors="coerce").dropna()
        if not timestamps.empty:
            return timestamps.max().isoformat()
    return fallback.isoformat()


def _block_timestamp_key(nodes_df: pd.DataFrame) -> str | None:
    if nodes_df.empty or "bloque_15min" not in nodes_df.columns:
        return None
    timestamps = pd.to_datetime(nodes_df["bloque_15min"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.max().isoformat()


def _is_duplicate_capture(capture_timestamp: str) -> bool:
    with _history_lock:
        return capture_timestamp in _seen_capture_timestamps


def _remember_capture_timestamp(capture_timestamp: str) -> None:
    with _history_lock:
        _seen_capture_timestamps.add(capture_timestamp)


def _scan_existing_capture_timestamps(parsed_dir: Path) -> set[str]:
    timestamps: set[str] = set()
    if not parsed_dir.exists():
        return timestamps

    for path in sorted(parsed_dir.glob("pm_parsed_*.csv")):
        try:
            frame = pd.read_csv(path, nrows=25)
        except Exception:
            continue
        if "fecha_hora" not in frame.columns:
            continue
        key = _timestamp_from_series(frame["fecha_hora"])
        if key is not None:
            timestamps.add(key)
    return timestamps


def _latest_block_timestamp(blocks_dir: Path) -> str | None:
    if not blocks_dir.exists():
        return None

    for path in sorted(blocks_dir.glob("nodes_15min_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            frame = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if "bloque_15min" not in frame.columns:
            continue
        key = _timestamp_from_series(frame["bloque_15min"])
        if key is not None:
            return key
    return None


def _latest_capture_timestamp_from_files(parsed_dir: Path) -> str | None:
    if not parsed_dir.exists():
        return None

    for path in sorted(parsed_dir.glob("pm_parsed_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            frame = pd.read_csv(path, nrows=25)
        except Exception:
            continue
        if "fecha_hora" not in frame.columns:
            continue
        key = _timestamp_from_series(frame["fecha_hora"])
        if key is not None:
            return key
    return None


def _timestamp_from_series(values: pd.Series) -> str | None:
    timestamps = pd.to_datetime(values, errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.max().isoformat()


def _latest_timestamp(values: set[str]) -> str | None:
    if not values:
        return None
    timestamps = pd.to_datetime(list(values), errors="coerce").dropna()
    if timestamps.empty:
        return sorted(values)[-1]
    return timestamps.max().isoformat()


def _predict_with_real_or_mock(nodes_df: pd.DataFrame, blocks_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths_cfg = CONFIG.get("paths", {})
    model_dir = resolve_path(PROJECT_ROOT, paths_cfg.get("selected_model_dir", "realtime_app/artifacts/selected_model"))

    result = predict_real_temporal(
        current_entities=nodes_df,
        project_root=PROJECT_ROOT,
        model_dir=model_dir,
        blocks_dir=blocks_dir,
    )
    if result.predictions is not None:
        return result.predictions, result.status

    fallback = predict_next_15min(nodes_df)
    fallback_mode = result.status.get("mode") or HISTORY_INSUFFICIENT_MODE
    fallback["modo_modelo"] = fallback_mode
    fallback["predictor"] = _fallback_predictor_name(fallback_mode)
    return fallback, result.status


def _fallback_predictor_name(mode: str) -> str:
    if mode == HISTORY_INSUFFICIENT_MODE:
        return "mock_identity_historico_insuficiente"
    if mode == FALLBACK_ERROR_MODE:
        return "mock_identity_error_inferencia"
    return "mock_identity_fallback"


def _payload(snapshot: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "metadata": _metadata(snapshot),
        "data": _records(snapshot[key]),
    }


def _metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    model_status = snapshot.get("model_status") or _current_model_status()
    return {
        "mode": CONFIG.get("mode", "demo"),
        "predictor_mode": model_status.get("mode", PREDICTOR_MODE),
        "model_loaded": bool(model_status.get("model_loaded")),
        "model_status": model_status,
        "mapping_mode": MAPPING_STATE.mode,
        "can_aggregate_by_node": MAPPING_STATE.can_aggregate,
        "updated_at": _iso(snapshot.get("updated_at")),
        "source_url": CONFIG.get("source", {}).get("xml_url"),
        "source_from_local_capture": bool(snapshot.get("from_cache")),
        "capture_path": snapshot.get("capture_path"),
        "parsed_path": snapshot.get("parsed_path"),
        "blocks_path": snapshot.get("blocks_path"),
        "predictions_path": snapshot.get("predictions_path"),
        "row_count": int(len(snapshot.get("raw", []))),
        "node_count": int(len(snapshot.get("nodes", []))),
        "notes": MAPPING_STATE.notes,
    }


def _current_model_status() -> dict[str, Any]:
    paths_cfg = CONFIG.get("paths", {})
    model_dir = resolve_path(PROJECT_ROOT, paths_cfg.get("selected_model_dir", "realtime_app/artifacts/selected_model"))
    blocks_dir = resolve_path(PROJECT_ROOT, paths_cfg.get("blocks_15min_dir", "realtime_app/data/blocks_15min"))
    return get_model_status(PROJECT_ROOT, model_dir, blocks_dir)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
