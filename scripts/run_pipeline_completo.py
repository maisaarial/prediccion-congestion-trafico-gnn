from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime


def escribir_log(log_path: Path, texto: str) -> None:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(texto)


def run(
        cmd: list[str],
        log_path: Path,
        run_id: str,
        results_dir: Path,
        git_push_interval_minutes: int = 0,
    ) -> None:

    separador = "\n" + "=" * 80 + "\n"

    print(separador)
    print("Ejecutando:", " ".join(cmd))
    print("=" * 80)

    escribir_log(log_path, separador)
    escribir_log(log_path, "Ejecutando: " + " ".join(cmd) + "\n")
    escribir_log(log_path, "=" * 80 + "\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    ultimo_push = time.time()
    intervalo_push = git_push_interval_minutes * 60

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            escribir_log(log_path, line)
            if git_push_interval_minutes > 0:
                ahora = time.time()
                if ahora - ultimo_push >= intervalo_push:
                    git_commit_push_parcial(run_id, log_path, results_dir)
                    ultimo_push = ahora

    process.wait()

    if process.returncode != 0:
        escribir_log(
            log_path,
            f"\nERROR: el comando terminó con código {process.returncode}\n",
        )
        raise subprocess.CalledProcessError(process.returncode, cmd)


def obtener_ultimo_mes(months: list[str]) -> str:
    meses_ordenados = sorted(
        months,
        key=lambda m: datetime.strptime(m, "%m-%Y")
    )
    return meses_ordenados[-1]


def obtener_meses_disponibles(traffic_dir: str = "data/raw/trafico") -> list[str]:
    traffic_path = Path(traffic_dir)
    archivos = sorted(traffic_path.glob("*.csv"))
    return [archivo.stem for archivo in archivos]


def validar_sensor_ultimo_mes(
    months: list[str],
    sensors_dir: str = "data/raw/sensores",
) -> None:
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


def validar_graphml(graphml_path: str) -> None:
    path = Path(graphml_path)

    if not path.exists():
        print(
            "\nAVISO: No se encontró el archivo graphml para casos de sentido:\n"
            f"{path}\n"
            "Si no usas --skip_sentido, los casos proximidad_sentido_v1 "
            "y proximidad_sentido_v2 no se podrán construir.\n"
        )
    else:
        print(f"Archivo graphml encontrado: {path}")

def expandir_casos_por_cluster_targets(casos: list[str], targets: list[str]) -> list[str]:
    casos_expandidos = []

    casos_con_cluster = {
        "proximidad",
        "proximidad_comportamiento",
        "proximidad_sentido_v1",
        "proximidad_sentido_v2",
    }

    for caso in casos:
        if caso == "sensores_tal_cual":
            casos_expandidos.append(caso)

        elif caso in casos_con_cluster:
            for target in targets:
                casos_expandidos.append(f"{caso}_{target}")

        else:
            casos_expandidos.append(caso)

    return casos_expandidos


def git_commit_push(run_id: str) -> None:
    subprocess.run(["git", "add", "."], check=True)

    commit_msg = f"experimento: {run_id}"

    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("No se hizo commit. Puede que no haya cambios.")
        print(result.stdout)
        print(result.stderr)
        return

    subprocess.run(["git", "push"], check=True)


def git_commit_push_parcial(run_id: str, log_path: Path, results_dir: Path) -> None:
    try:
        subprocess.run(["git", "add", str(log_path)], check=False)
        subprocess.run(["git", "add", str(results_dir)], check=False)

        commit_msg = f"avance experimento: {run_id}"

        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            print("No hay cambios nuevos para commitear en este avance.")
            return

        subprocess.run(["git", "push"], check=True)
        print(f"Avance subido a GitHub: {run_id}")

    except Exception as exc:
        print(f"No se pudo hacer push parcial: {exc}")


def git_push_periodico(run_id, log_path, results_dir, intervalo_minutos, stop_event):
    intervalo_segundos = intervalo_minutos * 60

    while not stop_event.wait(intervalo_segundos):
        print(f"\nPush periódico automático: {run_id}")

        try:
            subprocess.run(["git", "add", str(log_path)], check=False)
            subprocess.run(["git", "add", str(results_dir)], check=False)

            result = subprocess.run(
                ["git", "commit", "-m", f"avance experimento: {run_id}"],
                text=True,
                capture_output=True,
            )

            if result.returncode == 0:
                subprocess.run(["git", "push"], check=False)
                print("Push periódico completado.")
            else:
                print("No hay cambios nuevos para subir.")

        except Exception as exc:
            print(f"No se pudo hacer push periódico: {exc}")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo local: genera datasets y entrena modelos."
    )

    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--all_months", action="store_true")
    parser.add_argument("--skip_sentido", action="store_true")

    parser.add_argument(
        "--graphml_path",
        default="data/raw/osm/madrid_drive.graphml",
        help="Ruta al grafo vial graphml usado para construir casos de sentido.",
    )

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

    parser.add_argument(
        "--cluster_targets",
        nargs="+",
        default=["50", "500"],
        help="Número objetivo de clusters: 50 500"
    )

    parser.add_argument("--git_push", action="store_true")
    parser.add_argument(
        "--git_push_interval_minutes",
        type=int,
        default=0,
        help="Si es mayor que 0, hace commit/push automático cada X minutos."
    )

    args = parser.parse_args()

    casos_entrenamiento = expandir_casos_por_cluster_targets(
        casos=args.casos,
        targets=args.cluster_targets,
    )

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_path = log_dir / f"{run_id}.txt"

    results_dir = Path("results") / "experimentos_epocas_datos" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    inicio_total = time.time()
    fecha_inicio = datetime.now()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("INICIO PIPELINE\n")
        log.write(f"Fecha inicio: {fecha_inicio}\n")
        log.write(f"Comando: {' '.join(sys.argv)}\n")
        log.write("=" * 80 + "\n")

    print(f"\nLog guardándose en: {log_path}")

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

    validar_graphml(args.graphml_path)

    escribir_log(log_path, f"Meses usados: {months}\n")
    escribir_log(log_path, f"Casos: {casos_entrenamiento}\n")
    escribir_log(log_path, f"Adyacencias: {args.adyacencias}\n")
    escribir_log(log_path, f"Modelos: {args.modelos}\n")
    escribir_log(log_path, f"Fracciones: {args.fracciones}\n")
    escribir_log(log_path, f"Epochs: {args.epochs}\n")
    escribir_log(log_path, f"Batch size: {args.batch_size}\n")
    escribir_log(log_path, f"GraphML: {args.graphml_path}\n")

    gen_cmd = [
        sys.executable,
        "scripts/generar_datasets.py",
        "--casos",
        *args.casos,
        "--adyacencias",
        *args.adyacencias,
        "--months",
        *months,
        "--graphml_path",
        args.graphml_path,
        "--cluster_targets",
        *args.cluster_targets,
    ]

    if args.all_months:
        gen_cmd += ["--all_months"]

    if args.skip_sentido:
        gen_cmd += ["--skip_sentido"]

    train_cmd = [
        sys.executable,
        "scripts/probar_epocas_datos.py",
        "--output_dir",
        str(results_dir),
        "--casos",
        *casos_entrenamiento,
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

    stop_event = threading.Event()
    push_thread = None

    if args.git_push and args.git_push_interval_minutes > 0:
        push_thread = threading.Thread(
            target=git_push_periodico,
            args=(
                run_id,
                log_path,
                results_dir,
                args.git_push_interval_minutes,
                stop_event,
            ),
            daemon=True,
        )
        push_thread.start()

    try:
        run(
            gen_cmd,
            log_path,
            run_id,
            results_dir,
            args.git_push_interval_minutes,
        )

        run(
            train_cmd,
            log_path,
            run_id,
            results_dir,
            args.git_push_interval_minutes,
        )

    finally:
        fin_total = time.time()
        fecha_fin = datetime.now()
        duracion = fin_total - inicio_total

        if push_thread is not None:
            stop_event.set()
            push_thread.join(timeout=5)

        resumen_final = (
            "\n" + "=" * 80 + "\n"
            f"Fecha inicio: {fecha_inicio}\n"
            f"Fecha fin: {fecha_fin}\n"
            f"Tiempo total: {duracion:.2f} segundos ({duracion/60:.2f} minutos)\n"
            f"Log completo: {log_path}\n"
            + "=" * 80 + "\n"
        )

        print(resumen_final)
        escribir_log(log_path, resumen_final + f"\nDirectorio resultados: {results_dir}\n")
        
        if args.git_push:
            git_commit_push(run_id)


if __name__ == "__main__":
    main()