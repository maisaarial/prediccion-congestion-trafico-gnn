from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Ejecutando:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def obtener_ultimo_mes(months: list[str]) -> str:
    meses_ordenados = sorted(
        months,
        key=lambda m: datetime.strptime(m, "%m-%Y")
    )
    return meses_ordenados[-1]


def obtener_meses_disponibles(traffic_dir: str = "data/raw/trafico") -> list[str]:
    traffic_path = Path(traffic_dir)

    archivos = sorted(traffic_path.glob("*.csv"))

    meses = [
        archivo.stem
        for archivo in archivos
    ]

    return meses


def validar_sensor_ultimo_mes(months: list[str], sensors_dir: str = "data/raw/sensores") -> None:
    ultimo_mes = obtener_ultimo_mes(months)

    sensors_path = Path(sensors_dir) / f"pmed_ubicacion_{ultimo_mes}.csv"

    if not sensors_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de sensores del último mes:\n"
            f"{sensors_path}\n\n"
            f"Debes tener un archivo con este nombre:\n"
            f"pmed_ubicacion_{ultimo_mes}.csv"
        )

    print(f"Usando sensores del último mes seleccionado: {ultimo_mes}")
    print(f"Archivo sensores: {sensors_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo local: genera datasets y entrena modelos."
    )

    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--all_months", action="store_true")
    parser.add_argument("--skip_sentido", action="store_true")

    parser.add_argument(
        "--casos",
        nargs="+",
        default=[
            "sensores_tal_cual",
            "proximidad",
            "proximidad_comportamiento",
            "proximidad_sentido_v1",
            "proximidad_sentido_v2",
        ],
    )

    parser.add_argument(
        "--adyacencias",
        nargs="+",
        default=["cercania", "correlacion"],
    )

    parser.add_argument(
        "--modelos",
        nargs="+",
        default=["GCN_LSTM", "GCN_GRU", "GAT_LSTM"],
    )

    parser.add_argument("--fracciones", nargs="+", default=["0.25"])
    parser.add_argument("--epochs", nargs="+", default=["5"])
    parser.add_argument("--batch_size", default="8")

    args = parser.parse_args()

    if args.all_months:
        months = obtener_meses_disponibles("data/raw/trafico")
        if not months:
            raise FileNotFoundError(
                "No se encontraron CSV de tráfico en data/raw/trafico/"
            )
    else:
        if not args.months:
            raise ValueError(
                "Debes indicar --months 01-2025 o usar --all_months"
            )
        months = args.months

    validar_sensor_ultimo_mes(
        months=months,
        sensors_dir="data/raw/sensores",
    )

    gen_cmd = [
        sys.executable,
        "scripts/generar_datasets.py",
        "--casos",
        *args.casos,
        "--adyacencias",
        *args.adyacencias,
        "--months",
        *months,
    ]

    if args.all_months:
        gen_cmd += ["--all_months"]

    if args.skip_sentido:
        gen_cmd += ["--skip_sentido"]

    train_cmd = [
        sys.executable,
        "scripts/probar_epocas_datos.py",
        "--casos",
        *args.casos,
        "--adyacencias",
        *args.adyacencias,
        "--modelos",
        *args.modelos,
        "--fracciones",
        *args.fracciones,
        "--epochs",
        *args.epochs,
        "--batch_size",
        args.batch_size,
    ]

    run(gen_cmd)
    run(train_cmd)


if __name__ == "__main__":
    main()