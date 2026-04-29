from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_traffic_csv(path: str | Path, usecols: list[str] | None = None, sep: str = ";", encoding: str = "latin-1") -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, sep=sep, encoding=encoding)


def read_sensors_csv(
    path: str | Path,
    usecols: list[str] | None = None,
    sep: str = ";",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, sep=sep, encoding=encoding)


def find_traffic_files(traffic_dir: str | Path, months: list[str] | None = None, use_all_months: bool = False) -> list[Path]:
    """Busca CSV de tráfico por mes.

    Admite nombres como 01-2025.csv, 01-2025_algo.csv, trafico_01-2025.csv,
    o subcarpetas dentro de data/raw/trafico.
    """
    traffic_dir = Path(traffic_dir)
    if not traffic_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de tráfico: {traffic_dir}")

    all_csv = sorted(traffic_dir.rglob("*.csv"))
    if use_all_months:
        files = all_csv
    else:
        months = months or []
        files = []
        for month in months:
            matched = [p for p in all_csv if month in p.stem or month in p.name]
            if not matched:
                candidate = traffic_dir / f"{month}.csv"
                if candidate.exists():
                    matched = [candidate]
            files.extend(matched)
        # Quitar duplicados conservando orden
        seen = set()
        files = [p for p in files if not (p in seen or seen.add(p))]

    if not files:
        raise FileNotFoundError(
            f"No se encontraron CSV de tráfico en {traffic_dir}. "
            "Guarda tus archivos en data/raw/trafico/ y usa --months o --all_months."
        )
    return files


def load_traffic_files(
    traffic_dir: str | Path,
    months: list[str] | None = None,
    use_all_months: bool = False,
    usecols: list[str] | None = None,
    sep: str = ";",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    files = find_traffic_files(traffic_dir, months=months, use_all_months=use_all_months)
    dfs = []
    for file in files:
        print(f"Leyendo tráfico: {file}")
        dfs.append(read_traffic_csv(file, usecols=usecols, sep=sep, encoding=encoding))
    return pd.concat(dfs, ignore_index=True)
