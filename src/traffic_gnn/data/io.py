from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_traffic_csv(path: str | Path, usecols: list[str] | None = None, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, sep=sep)


def read_sensors_csv(
    path: str | Path,
    usecols: list[str] | None = None,
    sep: str = ";",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, sep=sep, encoding=encoding)
