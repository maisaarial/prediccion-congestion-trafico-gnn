from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None


DEFAULT_API_URL = os.environ.get("REALTIME_API_URL", "http://localhost:8000")
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "logo_deusto.png"

CATEGORY_ORDER = ["sin_dato", "bajo", "medio", "alto", "critico"]
CATEGORY_LABELS = {
    "sin_dato": "Sin dato",
    "bajo": "0 % a 30 %",
    "medio": "30 % a 60 %",
    "alto": "60 % a 80 %",
    "critico": "80 % a 100 %",
}
CATEGORY_STATUS = {
    "sin_dato": "Sin dato",
    "bajo": "Bajo",
    "medio": "Medio",
    "alto": "Alto",
    "critico": "Crítico",
}
CATEGORY_COLORS = {
    "sin_dato": [244, 246, 248, 230],
    "bajo": [0, 132, 61, 220],
    "medio": [47, 128, 237, 220],
    "alto": [0, 59, 122, 230],
    "critico": [218, 41, 28, 235],
}
CATEGORY_LINE_COLORS = {
    "sin_dato": [31, 41, 51, 220],
    "bajo": [255, 255, 255, 210],
    "medio": [255, 255, 255, 210],
    "alto": [255, 255, 255, 210],
    "critico": [255, 255, 255, 210],
}
CATEGORY_HEX = {
    "sin_dato": "#F4F6F8",
    "bajo": "#00843D",
    "medio": "#2F80ED",
    "alto": "#003B7A",
    "critico": "#DA291C",
}
FILTER_LABELS = {
    "sin_dato": "Sin dato",
    "bajo": "Bajo 0 % a 30 %",
    "medio": "Medio 30 % a 60 %",
    "alto": "Alto 60 % a 80 %",
    "critico": "Crítico 80 % a 100 %",
}


st.set_page_config(
    page_title="Predicción congestión Madrid",
    layout="wide",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --deusto-blue: #003B7A;
          --tech-blue: #2F80ED;
          --euskadi-green: #00843D;
          --euskadi-red: #DA291C;
          --app-bg: #F4F6F8;
          --text: #1F2933;
          --white: #FFFFFF;
        }

        .stApp {
          background: var(--app-bg);
          color: var(--text);
        }

        .main .block-container {
          padding-top: 1rem;
          padding-bottom: 1.5rem;
          max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
          color: var(--deusto-blue);
          letter-spacing: 0;
        }

        .deusto-header {
          display: flex;
          align-items: center;
          gap: 18px;
          background: var(--white);
          border-left: 6px solid var(--deusto-blue);
          border-radius: 10px;
          box-shadow: 0 8px 22px rgba(31, 41, 51, 0.10);
          padding: 16px 20px;
          margin-bottom: 12px;
        }

        .deusto-header img {
          width: 110px;
          max-height: 74px;
          object-fit: contain;
        }

        .deusto-title {
          margin: 0;
          color: var(--deusto-blue);
          font-size: 2rem;
          line-height: 1.15;
          font-weight: 750;
        }

        .deusto-subtitle {
          margin: 6px 0 0 0;
          color: var(--text);
          font-size: 1.02rem;
        }

        .metric-card {
          background: var(--white);
          border-radius: 10px;
          box-shadow: 0 6px 18px rgba(31, 41, 51, 0.09);
          border: 1px solid rgba(31, 41, 51, 0.08);
          border-top: 4px solid var(--deusto-blue);
          padding: 12px 13px;
          min-height: 86px;
        }

        .metric-card-label {
          color: var(--text);
          font-size: 0.82rem;
          line-height: 1.2;
          margin-bottom: 8px;
        }

        .metric-card-value {
          color: var(--deusto-blue);
          font-size: 1.45rem;
          font-weight: 760;
          line-height: 1;
        }

        div[data-testid="stAlert"] {
          border-radius: 10px;
          border: 1px solid rgba(0, 59, 122, 0.16);
          box-shadow: 0 4px 14px rgba(31, 41, 51, 0.06);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--white);
          border-radius: 10px;
          box-shadow: 0 6px 18px rgba(31, 41, 51, 0.08);
        }

        section[data-testid="stSidebar"] {
          background: #FFFFFF;
        }

        .block-container div[data-testid="stDataFrame"] {
          border-radius: 10px;
          overflow: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    logo_html = ""
    if LOGO_PATH.exists():
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_html = f'<img src="data:image/png;base64,{encoded_logo}" alt="Universidad de Deusto" />'

    st.markdown(
        f"""
        <div class="deusto-header">
          {logo_html}
          <div>
            <h1 class="deusto-title">Predicción de la congestión vehicular - Madrid</h1>
            <p class="deusto-subtitle">Proyecto de Fin de Máster - María Isabel Aristizabal</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def api_get(base_url: str, path: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def cached_api_get(base_url: str, path: str) -> dict[str, Any]:
    return api_get(base_url, path)


def classify_prediction(value: float) -> str:
    if pd.isna(value):
        return "sin_dato"
    value = float(value)
    if value < 0.30:
        return "bajo"
    if value < 0.60:
        return "medio"
    if value < 0.80:
        return "alto"
    return "critico"


def congestion_color(value: float) -> list[int]:
    return CATEGORY_COLORS[classify_prediction(value)]


def congestion_line_color(value: float) -> list[int]:
    return CATEGORY_LINE_COLORS[classify_prediction(value)]


def congestion_status(value: float) -> str:
    return CATEGORY_STATUS[classify_prediction(value)]


def prediction_range_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "prediccion_15min" not in df.columns:
        return {category: 0 for category in CATEGORY_ORDER}

    categories = pd.to_numeric(df["prediccion_15min"], errors="coerce").map(classify_prediction)
    counts = categories.value_counts().to_dict()
    return {category: int(counts.get(category, 0)) for category in CATEGORY_ORDER}


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "Sin dato"
    return f"{float(value) * 100:.1f} %"


def format_percentage_points(value: float) -> str:
    if pd.isna(value):
        return "Sin dato"
    return f"{float(value) * 100:.1f} pp"


def prepare_map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    map_df = df.dropna(subset=["lat", "lon"]).copy()
    if map_df.empty:
        return map_df

    sensor_count = map_df["sensor_count"] if "sensor_count" in map_df.columns else pd.Series(0, index=map_df.index)
    map_df["categoria_prediccion"] = map_df["prediccion_15min"].map(classify_prediction)
    map_df["color"] = map_df["prediccion_15min"].map(congestion_color)
    map_df["line_color"] = map_df["prediccion_15min"].map(congestion_line_color)
    map_df["radius"] = map_df["entity_type"].map({"node": 95, "sensor": 55}).fillna(65)
    map_df["status"] = map_df["prediccion_15min"].map(congestion_status)
    map_df["estado_operativo"] = map_df["categoria_prediccion"].map(
        lambda category: "sin_dato" if category == "sin_dato" else "activo"
    )
    map_df["congestion_actual_fmt"] = map_df["congestion_actual"].map(format_percent)
    map_df["prediccion_15min_fmt"] = map_df["prediccion_15min"].map(format_percent)
    map_df["cantidad_sensores_activos"] = pd.to_numeric(sensor_count, errors="coerce").fillna(0).astype(int)
    if "modo_modelo" in map_df.columns:
        map_df["modo_inferencia"] = map_df["modo_modelo"].fillna("DEMO_SIN_MODELO_REAL")
    else:
        map_df["modo_inferencia"] = "DEMO_SIN_MODELO_REAL"
    return map_df


def filter_map_dataframe(
    map_df: pd.DataFrame,
    selected_categories: list[str],
    selected_node_id: str,
) -> tuple[pd.DataFrame, bool]:
    if map_df.empty:
        return map_df, False

    filtered = map_df[map_df["categoria_prediccion"].isin(selected_categories)].copy()
    selected = map_df[map_df["node_id"].astype(str) == str(selected_node_id)].copy()
    selected_outside_filter = (
        not selected.empty
        and str(selected.iloc[0]["categoria_prediccion"]) not in selected_categories
    )

    if selected_outside_filter:
        filtered = pd.concat([filtered, selected], ignore_index=True)
        filtered = filtered.drop_duplicates(subset=["node_id"], keep="last")

    return filtered.reset_index(drop=True), selected_outside_filter


def render_legend(counts: dict[str, int]) -> None:
    legend_items = []
    for category in CATEGORY_ORDER:
        label = CATEGORY_LABELS[category]
        count = counts.get(category, 0)
        text = f"{label}: {count} nodos"
        border = "1px solid #1F2933" if category == "sin_dato" else "1px solid transparent"
        legend_items.append(
            f'<span><span style="display:inline-block;width:12px;height:12px;'
            f'background:{CATEGORY_HEX[category]};border:{border};border-radius:50%;margin-right:5px;"></span>{text}</span>'
        )

    st.markdown(
        f"""
        <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:6px 0 12px 0;font-size:0.92rem;color:#1F2933;">
          {''.join(legend_items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: object, accent: str = "#003B7A") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{accent};">
          <div class="metric-card-label">{label}</div>
          <div class="metric-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(map_df: pd.DataFrame, counts: dict[str, int], history_status: dict[str, Any]) -> None:
    available_blocks = int(history_status.get("available_blocks", 0) or 0)
    cards = [
        ("Total nodos", len(map_df), "#003B7A"),
        ("Bajo 0 % a 30 %", counts["bajo"], "#00843D"),
        ("Medio 30 % a 60 %", counts["medio"], "#2F80ED"),
        ("Alto 60 % a 80 %", counts["alto"], "#003B7A"),
        ("Crítico 80 % a 100 %", counts["critico"], "#DA291C"),
        ("Sin dato", counts["sin_dato"], "#1F2933"),
        ("Datos históricos disponibles", available_blocks, "#2F80ED"),
    ]

    columns = st.columns(len(cards))
    for column, (label, value, accent) in zip(columns, cards):
        with column:
            render_metric_card(label, value, accent)


def render_category_filter() -> list[str]:
    st.markdown("**Rangos visibles en el mapa**")
    cols = st.columns(len(CATEGORY_ORDER))
    selected_categories: list[str] = []

    for index, category in enumerate(CATEGORY_ORDER):
        with cols[index]:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:6px;min-height:24px;">
                  <span style="display:inline-block;width:11px;height:11px;background:{CATEGORY_HEX[category]};border:{'1px solid #1F2933' if category == 'sin_dato' else '1px solid transparent'};border-radius:50%;"></span>
                  <span style="font-size:0.86rem;color:#1F2933;line-height:1.1;">{FILTER_LABELS[category]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            checked = st.checkbox(
                FILTER_LABELS[category],
                value=True,
                key=f"map_filter_{category}",
                label_visibility="collapsed",
            )
            if checked:
                selected_categories.append(category)

    if not selected_categories:
        st.warning("No hay rangos activos. Activa al menos una categoria para ver nodos filtrados.")

    return selected_categories


def extract_clicked_node_id(chart_state: Any) -> str | None:
    # TODO: Tighten this parser if Streamlit changes the PydeckState selection schema.
    def find_node_id(value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif not isinstance(value, (dict, list, tuple, str, int, float, bool)):
            try:
                value = dict(value)
            except Exception:
                return None

        if isinstance(value, dict):
            if value.get("node_id") is not None:
                return str(value["node_id"])
            for item in value.values():
                found = find_node_id(item)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = find_node_id(item)
                if found is not None:
                    return found
        return None

    return find_node_id(chart_state)


def render_map(map_df: pd.DataFrame, counts: dict[str, int], selected_node_id: str) -> str | None:
    if map_df.empty:
        st.info("No hay coordenadas disponibles para pintar el mapa.")
        return None

    if pdk is None:
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), height=740)
        render_legend(counts)
        return None

    base_df = map_df[map_df["node_id"].astype(str) != str(selected_node_id)].copy()
    selected_df = map_df[map_df["node_id"].astype(str) == str(selected_node_id)].copy()

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            id="traffic_nodes",
            data=base_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            stroked=True,
            get_line_color="line_color",
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    ]

    if not selected_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="selected_node",
                data=selected_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=180,
                pickable=True,
                stroked=True,
                get_line_color=[0, 0, 0, 255],
                line_width_min_pixels=4,
            )
        )

    view_state = pdk.ViewState(
        latitude=40.4168,
        longitude=-3.7038,
        zoom=10.5,
        pitch=0,
    )
    tooltip = {
        "html": (
            "<b>{descripcion}</b><br/>"
            "Nodo GNN: {node_id}<br/>"
            "Estado: {estado_operativo}<br/>"
            "Predicción de congestión 15 min: {prediccion_15min_fmt}<br/>"
            "Congestión actual: {congestion_actual_fmt}<br/>"
            "Sensores activos: {cantidad_sensores_activos}<br/>"
            "Modo: {modo_inferencia}"
        ),
        "style": {"fontFamily": "Arial", "fontSize": "12px"},
    }
    chart_state = st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
        ),
        use_container_width=True,
        height=740,
        selection_mode="single-object",
        on_select="rerun",
        key="traffic_map",
    )
    render_legend(counts)
    return extract_clicked_node_id(chart_state)


def build_nodes_display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["congestion_actual"] = out["congestion_actual"].map(format_percent)
    out["prediccion_15min"] = out["prediccion_15min"].map(format_percent)
    out["diferencia"] = out["diferencia"].map(format_percentage_points)
    return out


def build_observations_display_table(observations: pd.DataFrame) -> pd.DataFrame:
    out = observations.copy()
    if "congestion_demo" in out.columns:
        out["congestion_demo"] = pd.to_numeric(out["congestion_demo"], errors="coerce").map(format_percent)
    return out


with st.sidebar:
    api_base_url = st.text_input("Backend API", value=DEFAULT_API_URL)
    if st.button("Actualizar", use_container_width=True):
        cached_api_get.clear()

inject_custom_css()
render_header()

try:
    health = cached_api_get(api_base_url, "/health")
    predictions_payload = cached_api_get(api_base_url, "/predictions")
    model_status = cached_api_get(api_base_url, "/model/status")
    history_status = cached_api_get(api_base_url, "/history/status")
except Exception as exc:
    st.error(f"No se pudo conectar con el backend: {exc}")
    st.stop()

records = predictions_payload.get("data", [])
metadata = predictions_payload.get("metadata", {})
df = pd.DataFrame(records)

active_mode = model_status.get("mode") or metadata.get("predictor_mode")
if active_mode == "REAL_MODEL_TEMPORAL":
    technical_mode_message = (
        "Modelo GNN temporal activo: predicción real calculada con checkpoint de integración. "
        "No corresponde al modelo final."
    )
elif active_mode == "DEMO_HISTORICO_INSUFICIENTE":
    technical_mode_message = (
        "Modo demo: histórico insuficiente para inferencia real. "
        "Se muestra congestión actual como predicción provisional."
    )
else:
    technical_mode_message = (
        "Modo demo: la predicción 15 min es igual a la congestión actual. "
        "Todavía no usa el modelo GNN real."
    )

available_blocks = int(history_status.get("available_blocks", 0) or 0)
required_blocks = int(history_status.get("required_blocks", 12) or 12)
window_ready = bool(history_status.get("window_ready"))
collector_text = "Recolector automático activo" if history_status.get("auto_collector_enabled") else "Recolector automático inactivo"
history_ready_text = "Sí" if window_ready else "No"
technical_history_message = (
    f"{collector_text} · Bloques disponibles: {available_blocks}/{required_blocks} · "
    f"Histórico suficiente para modelo real: {history_ready_text}"
)

if df.empty:
    st.info("El backend esta activo, pero aun no hay datos para mostrar.")
    st.stop()

for column in ["congestion_actual", "prediccion_15min", "diferencia"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["status"] = df["prediccion_15min"].map(congestion_status)
sensor_count = df["sensor_count"] if "sensor_count" in df.columns else pd.Series(0, index=df.index)
df["cantidad_sensores_activos"] = pd.to_numeric(sensor_count, errors="coerce").fillna(0).astype(int)

map_df = prepare_map_dataframe(df)
full_range_counts = prediction_range_counts(map_df)
render_summary_metrics(map_df, full_range_counts, history_status)

node_options = df["node_id"].astype(str).tolist()
if "selected_node_id" not in st.session_state or st.session_state.selected_node_id not in node_options:
    st.session_state.selected_node_id = node_options[0]

selected_categories = render_category_filter()
selected_node_id = str(st.session_state.selected_node_id)
visible_map_df, selected_outside_filter = filter_map_dataframe(
    map_df,
    selected_categories=selected_categories,
    selected_node_id=selected_node_id,
)
visible_range_counts = prediction_range_counts(visible_map_df)

if selected_outside_filter:
    st.info("El nodo seleccionado esta fuera del filtro actual, pero se mantiene visible y resaltado en el mapa.")

map_col, detail_col = st.columns([3, 1], gap="medium")
with map_col:
    clicked_node_id = render_map(visible_map_df, visible_range_counts, selected_node_id)
    if clicked_node_id in node_options and clicked_node_id != st.session_state.selected_node_id:
        st.session_state.selected_node_id = clicked_node_id
        st.rerun()

with detail_col:
    with st.container(border=True):
        st.subheader("Detalle del nodo seleccionado")
        selected_node = st.selectbox(
            "Nodo o sensor",
            node_options,
            index=node_options.index(st.session_state.selected_node_id),
            key="selected_node_id",
        )
        st.markdown(f"**Nodo GNN seleccionado:** {selected_node}")
        detail = cached_api_get(api_base_url, f"/nodes/{selected_node}")
        node = detail.get("node", {})
        st.metric(
            "Predicción de congestión del nodo seleccionado a 15 min",
            format_percent(pd.to_numeric(node.get("prediccion_15min"), errors="coerce")),
        )
        st.metric(
            "Congestión actual del nodo seleccionado",
            format_percent(pd.to_numeric(node.get("congestion_actual"), errors="coerce")),
        )
        st.caption(node.get("descripcion", ""))

        st.subheader("Puntos de medida que componen el nodo seleccionado")
        observations = pd.DataFrame(detail.get("observations", []))
        if not observations.empty:
            visible_obs_cols = [
                column
                for column in ["idelem", "descripcion"]
                if column in observations.columns
            ]
            st.dataframe(observations[visible_obs_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No hay puntos de medida asociados al nodo seleccionado en la captura actual.")

st.subheader("Tabla de nodos")
visible_cols = [
    column
    for column in [
        "node_id",
        "entity_type",
        "descripcion",
        "status",
        "congestion_actual",
        "prediccion_15min",
        "diferencia",
        "intensidad",
        "ocupacion",
        "carga",
        "cantidad_sensores_activos",
    ]
    if column in df.columns
]
nodes_display = build_nodes_display_table(df)
st.dataframe(nodes_display[visible_cols], use_container_width=True, hide_index=True)

with st.expander("Estado técnico"):
    st.write(technical_mode_message)
    st.write(technical_history_message)
    st.json(metadata)
    st.json(model_status)
    st.json(history_status)
