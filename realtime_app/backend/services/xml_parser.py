from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree as ET

import pandas as pd


OUTPUT_COLUMNS = [
    "idelem",
    "descripcion",
    "intensidad",
    "ocupacion",
    "carga",
    "nivelServicio",
    "st_x",
    "st_y",
    "fecha_hora",
]

FIELD_ALIASES = {
    "idelem": ("idelem", "id", "id_elem", "idpunto"),
    "descripcion": ("descripcion", "description", "nombre"),
    "intensidad": ("intensidad",),
    "ocupacion": ("ocupacion",),
    "carga": ("carga",),
    "nivelServicio": ("nivelservicio", "nivel_servicio", "nivel"),
    "st_x": ("st_x", "x", "longitud", "lon"),
    "st_y": ("st_y", "y", "latitud", "lat"),
    "fecha_hora": ("fecha_hora", "fechahora", "fecha", "hora"),
}


def _normalize_tag(tag: str) -> str:
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag.strip().replace("-", "_").replace(" ", "_").lower()


def _clean_number(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None
    return value


def _children_as_dict(element: ET.Element) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for child in list(element):
        tag = _normalize_tag(child.tag)
        text = child.text.strip() if child.text else None
        values[tag] = text
    return values


def _first_value(values: dict[str, str | None], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        value = values.get(alias)
        if value not in (None, ""):
            return value
    return None


def _find_global_timestamp(root: ET.Element) -> str | None:
    root_values = _children_as_dict(root)
    return _first_value(root_values, FIELD_ALIASES["fecha_hora"])


def parse_pm_xml(xml_content: str, captured_at: datetime | None = None) -> pd.DataFrame:
    root = ET.fromstring(xml_content)
    global_timestamp = _find_global_timestamp(root)
    fallback_timestamp = captured_at.isoformat() if captured_at else None

    rows: list[dict[str, object]] = []
    for element in root.iter():
        values = _children_as_dict(element)
        if not _first_value(values, FIELD_ALIASES["idelem"]):
            continue

        row = {
            output_col: _first_value(values, FIELD_ALIASES[output_col])
            for output_col in OUTPUT_COLUMNS
        }
        row["fecha_hora"] = row["fecha_hora"] or global_timestamp or fallback_timestamp
        rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if df.empty:
        return df

    numeric_cols = ["idelem", "intensidad", "ocupacion", "carga", "nivelServicio", "st_x", "st_y"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column].map(_clean_number), errors="coerce")

    df["idelem"] = df["idelem"].astype("Int64")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce", dayfirst=True)

    if captured_at is not None:
        df["fecha_hora"] = df["fecha_hora"].fillna(pd.Timestamp(captured_at))

    return df.dropna(subset=["idelem"]).reset_index(drop=True)
