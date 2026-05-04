from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath("src"))

import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from traffic_gnn.models.gcn_lstm import GCN_LSTM
from traffic_gnn.models.gcn_gru import GCN_GRU
from traffic_gnn.models.gat_lstm import GAT_LSTM
from traffic_gnn.training.engine import train_one_epoch, evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODELOS = {
    "GCN_LSTM": GCN_LSTM,
    "GCN_GRU": GCN_GRU,
    "GAT_LSTM": GAT_LSTM,
}


CASOS = [
    "sensores_tal_cual",
    "proximidad",
    "proximidad_comportamiento",
    "proximidad_sentido_v1",
    "proximidad_sentido_v2",
]


TIPOS_ADYACENCIA = ["cercania", "correlacion"]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crear_subset(dataset, fraccion):
    n_total = len(dataset)
    n_usar = max(1, int(n_total * fraccion))
    return TensorDataset(dataset.tensors[0][:n_usar], dataset.tensors[1][:n_usar])


def cargar_caso(caso, tipo_adyacencia, data_dir):
    caso_dir = Path(data_dir) / caso / tipo_adyacencia

    train_data = torch.load(caso_dir / "train.pt", map_location="cpu", weights_only=False)
    val_data = torch.load(caso_dir / "val.pt", map_location="cpu", weights_only=False)
    test_data = torch.load(caso_dir / "test.pt", map_location="cpu", weights_only=False)
    graph_data = torch.load(caso_dir / "graph.pt", map_location="cpu", weights_only=False)

    return train_data, val_data, test_data, graph_data


def construir_modelo(nombre_modelo, graph_data, hidden_channels, lstm_hidden):
    model_cls = MODELOS[nombre_modelo]

    model = model_cls(
        num_nodes=int(graph_data["num_nodes"]),
        in_channels=int(graph_data["num_features"]),
        hidden_channels=hidden_channels,
        lstm_hidden=lstm_hidden,
    )

    return model.to(DEVICE)


def guardar_curva_entrenamiento(history, output_path, titulo):
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val loss")
    plt.plot(history["epoch"], history["val_mae"], label="Val MAE")
    plt.plot(history["epoch"], history["val_rmse"], label="Val RMSE")
    plt.title(titulo)
    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def guardar_pred_vs_real(
    y_pred,
    y_true,
    output_path,
    titulo,
    max_nodos=5,
    max_pasos=150,
):
    """
    Guarda gráficas de predicción vs real para algunos nodos representativos.

    Funciona también para sensores_tal_cual aunque tenga miles de nodos.
    No grafica todos los nodos, solo una muestra repartida.
    """

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(-1)

    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    if y_pred.ndim != 2 or y_true.ndim != 2:
        print(
            f"No se generan pred_vs_real para {titulo}: "
            f"shapes no válidos y_pred={y_pred.shape}, y_true={y_true.shape}"
        )
        return

    n_muestras, n_nodos = y_true.shape

    if n_muestras == 0 or n_nodos == 0:
        print(f"No se generan pred_vs_real para {titulo}: y_true vacío.")
        return

    pasos = min(max_pasos, n_muestras)

    if n_nodos <= max_nodos:
        nodos = list(range(n_nodos))
    else:
        nodos = np.linspace(0, n_nodos - 1, num=max_nodos, dtype=int).tolist()

    for nodo in nodos:
        plt.figure(figsize=(11, 5))

        plt.plot(
            range(pasos),
            y_true[:pasos, nodo],
            label="Real",
            linewidth=1.8,
        )

        plt.plot(
            range(pasos),
            y_pred[:pasos, nodo],
            label="Predicción",
            linewidth=1.8,
        )

        plt.title(f"{titulo} | Nodo {nodo}")
        plt.xlabel("Muestra temporal de test")
        plt.ylabel("Congestión")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        nodo_path = output_path.with_name(f"{output_path.stem}_nodo_{nodo}.png")
        plt.savefig(nodo_path, dpi=300)
        plt.close()

    print(
        f"Gráficas pred_vs_real guardadas para {titulo}: "
        f"{len(nodos)} nodos, {pasos} pasos."
    )


def guardar_comparaciones_globales(df, output_dir):
    output_dir = Path(output_dir)
    comp_dir = output_dir / "graficas" / "comparaciones"
    comp_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    for col, fname, title in [
        ("modelo", "mae_por_modelo.png", "MAE por modelo"),
        ("caso", "mae_por_caso.png", "MAE por caso"),
        ("tipo_adyacencia", "mae_por_adyacencia.png", "MAE por tipo de adyacencia"),
    ]:
        if col in df.columns:
            resumen = df.groupby(col)["test_mae"].mean().sort_values()

            plt.figure(figsize=(10, 5))
            resumen.plot(kind="bar")
            plt.title(title)
            plt.ylabel("Test MAE promedio")
            plt.xlabel(col)
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(comp_dir / fname, dpi=300)
            plt.close()

    plt.figure(figsize=(10, 6))

    for modelo in df["modelo"].unique():
        tmp = df[df["modelo"] == modelo]
        plt.scatter(
            tmp["epochs_ejecutadas"],
            tmp["test_mae"],
            label=modelo,
            s=80,
        )

    plt.title("Comparación de modelos: épocas ejecutadas vs MAE")
    plt.xlabel("Épocas ejecutadas")
    plt.ylabel("Test MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(comp_dir / "comparacion_modelos_mae.png", dpi=300)
    plt.close()


def ejecutar_experimento(
    caso,
    tipo_adyacencia,
    nombre_modelo,
    fraccion_datos,
    max_epochs,
    batch_size,
    patience,
    hidden_channels,
    lstm_hidden,
    data_dir,
    output_dir,
    nodos_pred_vs_real,
    max_pasos_pred_vs_real,
):
    print("\n" + "=" * 80)
    print(f"CASO: {caso}")
    print(f"ADYACENCIA: {tipo_adyacencia}")
    print(f"MODELO: {nombre_modelo}")
    print(f"DATOS: {int(fraccion_datos * 100)}%")
    print(f"ÉPOCAS MÁXIMAS: {max_epochs}")
    print("=" * 80)

    train_data, val_data, test_data, graph_data = cargar_caso(
        caso,
        tipo_adyacencia,
        data_dir,
    )

    train_subset = crear_subset(train_data, fraccion_datos)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    edge_index = graph_data["edge_index"].to(DEVICE)
    edge_weight = graph_data["edge_weight"].to(DEVICE)

    model = construir_modelo(
        nombre_modelo,
        graph_data,
        hidden_channels,
        lstm_hidden,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_sin_mejora = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }

    inicio = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            edge_index,
            edge_weight,
            DEVICE,
        )

        val_loss, val_mae, val_rmse, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            edge_index,
            edge_weight,
            DEVICE,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Val RMSE: {val_rmse:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if epochs_sin_mejora >= patience:
            print(f"Early stopping en época {epoch}")
            break

    tiempo_total = time.time() - inicio

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae, test_rmse, y_pred, y_true = evaluate(
        model,
        test_loader,
        criterion,
        edge_index,
        edge_weight,
        DEVICE,
    )

    print(f"\n⏱️ Tiempo experimento: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")

    exp_name = (
        f"{caso}_{tipo_adyacencia}_{nombre_modelo}_"
        f"datos_{int(fraccion_datos * 100)}_epochs_{max_epochs}"
    )

    output_dir = Path(output_dir)

    modelos_dir = output_dir / "modelos"
    curvas_dir = output_dir / "graficas" / "curvas_entrenamiento"
    pred_dir = output_dir / "graficas" / "pred_vs_real"
    historiales_dir = output_dir / "historiales"

    for d in [modelos_dir, curvas_dir, pred_dir, historiales_dir]:
        d.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), modelos_dir / f"{exp_name}.pt")

    pd.DataFrame(history).to_csv(
        historiales_dir / f"{exp_name}.csv",
        index=False,
    )

    guardar_curva_entrenamiento(
        history,
        curvas_dir / f"{exp_name}.png",
        exp_name,
    )

    guardar_pred_vs_real(
        y_pred=y_pred,
        y_true=y_true,
        output_path=pred_dir / f"{exp_name}.png",
        titulo=exp_name,
        max_nodos=nodos_pred_vs_real,
        max_pasos=max_pasos_pred_vs_real,
    )

    return {
        "caso": caso,
        "tipo_adyacencia": tipo_adyacencia,
        "modelo": nombre_modelo,
        "fraccion_datos": fraccion_datos,
        "porcentaje_datos": int(fraccion_datos * 100),
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "epochs_ejecutadas": len(history["epoch"]),
        "batch_size": batch_size,
        "hidden_channels": hidden_channels,
        "lstm_hidden": lstm_hidden,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "tiempo_segundos": tiempo_total,
        "modelo_guardado": str(modelos_dir / f"{exp_name}.pt"),
        "curva_guardada": str(curvas_dir / f"{exp_name}.png"),
        "pred_vs_real_dir": str(pred_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entrena y compara GCN_LSTM, GCN_GRU y GAT_LSTM por caso y adyacencia."
    )

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experimentos_epocas_datos")
    parser.add_argument("--casos", nargs="+", default=CASOS)
    parser.add_argument("--adyacencias", nargs="+", default=TIPOS_ADYACENCIA)
    parser.add_argument("--modelos", nargs="+", default=["GCN_LSTM", "GCN_GRU", "GAT_LSTM"])
    parser.add_argument("--fracciones", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--epochs", nargs="+", type=int, default=[20, 50, 100, 150, 200])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lstm_hidden", type=int, default=64)

    parser.add_argument(
        "--nodos_pred_vs_real",
        type=int,
        default=5,
        help="Número de nodos representativos a graficar en real vs predicción.",
    )

    parser.add_argument(
        "--max_pasos_pred_vs_real",
        type=int,
        default=150,
        help="Número máximo de pasos temporales de test a graficar.",
    )

    args = parser.parse_args()

    set_seed(42)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    resultados = []

    for caso in args.casos:
        for tipo_ady in args.adyacencias:
            for modelo in args.modelos:
                for fraccion in args.fracciones:
                    for max_epochs in args.epochs:
                        try:
                            res = ejecutar_experimento(
                                caso,
                                tipo_ady,
                                modelo,
                                fraccion,
                                max_epochs,
                                args.batch_size,
                                args.patience,
                                args.hidden_channels,
                                args.lstm_hidden,
                                args.data_dir,
                                args.output_dir,
                                args.nodos_pred_vs_real,
                                args.max_pasos_pred_vs_real,
                            )

                            resultados.append(res)

                            pd.DataFrame(resultados).to_csv(
                                Path(args.output_dir) / "resultados_completos.csv",
                                index=False,
                            )

                        except FileNotFoundError as e:
                            print(
                                f"\nArchivo no encontrado. Se omite: "
                                f"caso={caso}, adyacencia={tipo_ady}, modelo={modelo}. {e}"
                            )

                        except RuntimeError as e:
                            print(
                                f"\nERROR EN EXPERIMENTO: "
                                f"caso={caso}, adyacencia={tipo_ady}, modelo={modelo}. {e}"
                            )

                            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(
                                f"\nERROR no controlado: "
                                f"caso={caso}, adyacencia={tipo_ady}, modelo={modelo}. {e}"
                            )

    df = pd.DataFrame(resultados)

    if not df.empty:
        df_ordenado = df.sort_values(
            by=["test_mae", "test_rmse"],
            ascending=True,
        )

        print("\nTIEMPOS DE EJECUCIÓN (TOP 20):")
        print(
            df_ordenado[
                ["caso", "modelo", "tipo_adyacencia", "tiempo_segundos"]
            ].head(20)
        )

        df_ordenado.to_csv(
            Path(args.output_dir) / "ranking_modelos.csv",
            index=False,
        )

        guardar_comparaciones_globales(df_ordenado, args.output_dir)

        print("\nMEJORES RESULTADOS")
        print(df_ordenado.head(20))
        print(f"\nResultados guardados en: {args.output_dir}")

    else:
        print(
            "No se generaron resultados. "
            "Revisa data/processed o ejecuta scripts/generar_datasets.py primero."
        )


if __name__ == "__main__":
    main()