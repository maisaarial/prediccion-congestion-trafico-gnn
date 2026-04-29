from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Ejecutando:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo local: genera datasets y entrena modelos.")
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--all_months", action="store_true")
    parser.add_argument("--skip_sentido", action="store_true")
    parser.add_argument("--casos", nargs="+", default=["sensores_tal_cual", "proximidad", "proximidad_comportamiento", "proximidad_sentido_v1", "proximidad_sentido_v2"])
    parser.add_argument("--adyacencias", nargs="+", default=["cercania", "correlacion"])
    parser.add_argument("--modelos", nargs="+", default=["GCN_LSTM", "GCN_GRU", "GAT_LSTM"])
    parser.add_argument("--fracciones", nargs="+", default=["0.25"])
    parser.add_argument("--epochs", nargs="+", default=["5"])
    parser.add_argument("--batch_size", default="8")
    args = parser.parse_args()

    gen_cmd = [sys.executable, "scripts/generar_datasets.py", "--casos", *args.casos, "--adyacencias", *args.adyacencias]
    if args.months:
        gen_cmd += ["--months", *args.months]
    if args.all_months:
        gen_cmd += ["--all_months"]
    if args.skip_sentido:
        gen_cmd += ["--skip_sentido"]

    train_cmd = [
        sys.executable,
        "scripts/probar_epocas_datos.py",
        "--casos", *args.casos,
        "--adyacencias", *args.adyacencias,
        "--modelos", *args.modelos,
        "--fracciones", *args.fracciones,
        "--epochs", *args.epochs,
        "--batch_size", args.batch_size,
    ]

    run(gen_cmd)
    run(train_cmd)


if __name__ == "__main__":
    main()
