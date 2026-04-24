import os
import json
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


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELOS = {
    "GCN_LSTM": GCN_LSTM,
    "GCN_GRU": GCN_GRU,
    "GAT_LSTM": GAT_LSTM,
}


# Ajusta estos nombres a los casos que ya tienes guardados
CASOS = [
    "proximidad",
    "proximidad_comportamiento",
    "proximidad_sentido",
    "correlacion",
]


# ============================================================
# UTILIDADES
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crear_subset(dataset, fraccion):
    n_total = len(dataset)
    n_usar = max(1, int(n_total * fraccion))

    X = dataset.tensors[0][:n_usar]
    y = dataset.tensors[1][:n_usar]

    return TensorDataset(X, y)


def cargar_caso(caso, data_dir):
    """
    Espera esta estructura:

    data/processed/{caso}/train.pt
    data/processed/{caso}/val.pt
    data/processed/{caso}/test.pt
    data/processed/{caso}/graph.pt

    graph.pt debe tener:
    {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": N,
        "num_features": F,
        "seq_len": T
    }
    """

    caso_dir = Path(data_dir) / caso

    train_data = torch.load(caso_dir / "train.pt", map_location="cpu")
    val_data = torch.load(caso_dir / "val.pt", map_location="cpu")
    test_data = torch.load(caso_dir / "test.pt", map_location="cpu")
    graph_data = torch.load(caso_dir / "graph.pt", map_location="cpu")

    return train_data, val_data, test_data, graph_data


def construir_modelo(nombre_modelo, graph_data, hidden_channels, lstm_hidden):
    ModelClass = MODELOS[nombre_modelo]

    num_nodes = graph_data["num_nodes"]
    num_features = graph_data["num_features"]

    model = ModelClass(
        num_nodes=num_nodes,
        in_channels=num_features,
        hidden_channels=hidden_channels,
        lstm_hidden=lstm_hidden,
        out_channels=1,
    )

    return model.to(DEVICE)


def guardar_grafica(history, output_path, titulo):
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
    plt.show()


# ============================================================
# ENTRENAMIENTO DE UN EXPERIMENTO
# ============================================================

def ejecutar_experimento(
    caso,
    nombre_modelo,
    fraccion_datos,
    max_epochs,
    batch_size,
    patience,
    hidden_channels,
    lstm_hidden,
    data_dir,
    output_dir,
):

    print("\n" + "=" * 80)
    print(f"CASO: {caso}")
    print(f"MODELO: {nombre_modelo}")
    print(f"DATOS: {int(fraccion_datos * 100)}%")
    print(f"ÉPOCAS MÁXIMAS: {max_epochs}")
    print("=" * 80)

    train_data, val_data, test_data, graph_data = cargar_caso(caso, data_dir)

    train_subset = crear_subset(train_data, fraccion_datos)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    edge_index = graph_data["edge_index"].to(DEVICE)
    edge_weight = graph_data["edge_weight"].to(DEVICE)

    model = construir_modelo(
        nombre_modelo=nombre_modelo,
        graph_data=graph_data,
        hidden_channels=hidden_channels,
        lstm_hidden=lstm_hidden,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None
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
            criterion,
            optimizer,
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
            best_state = model.state_dict()
            best_epoch = epoch
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if epochs_sin_mejora >= patience:
            print(f"Early stopping en época {epoch}")
            break

    tiempo_total = time.time() - inicio

    model.load_state_dict(best_state)

    test_loss, test_mae, test_rmse, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        edge_index,
        edge_weight,
        DEVICE,
    )

    exp_name = f"{caso}_{nombre_modelo}_datos_{int(fraccion_datos * 100)}_epochs_{max_epochs}"

    modelos_dir = Path(output_dir) / "modelos"
    graficas_dir = Path(output_dir) / "graficas"
    historiales_dir = Path(output_dir) / "historiales"

    modelos_dir.mkdir(parents=True, exist_ok=True)
    graficas_dir.mkdir(parents=True, exist_ok=True)
    historiales_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), modelos_dir / f"{exp_name}.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(historiales_dir / f"{exp_name}.csv", index=False)

    guardar_grafica(
        history,
        graficas_dir / f"{exp_name}.png",
        titulo=exp_name,
    )

    resultado = {
        "caso": caso,
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
        "grafica_guardada": str(graficas_dir / f"{exp_name}.png"),
    }

    return resultado


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experimentos_epocas_datos")

    parser.add_argument("--casos", nargs="+", default=CASOS)
    parser.add_argument("--modelos", nargs="+", default=["GCN_LSTM", "GCN_GRU", "GAT_LSTM"])

    parser.add_argument("--fracciones", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--epochs", nargs="+", type=int, default=[20, 50, 100, 150, 200])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lstm_hidden", type=int, default=64)

    args = parser.parse_args()

    set_seed(42)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    resultados = []

    for caso in args.casos:
        for modelo in args.modelos:
            for fraccion in args.fracciones:
                for max_epochs in args.epochs:

                    try:
                        resultado = ejecutar_experimento(
                            caso=caso,
                            nombre_modelo=modelo,
                            fraccion_datos=fraccion,
                            max_epochs=max_epochs,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            hidden_channels=args.hidden_channels,
                            lstm_hidden=args.lstm_hidden,
                            data_dir=args.data_dir,
                            output_dir=args.output_dir,
                        )

                        resultados.append(resultado)

                        df_resultados = pd.DataFrame(resultados)
                        df_resultados.to_csv(
                            Path(args.output_dir) / "resultados_completos.csv",
                            index=False,
                        )

                    except RuntimeError as e:
                        print("\nERROR EN EXPERIMENTO")
                        print(f"Caso: {caso}")
                        print(f"Modelo: {modelo}")
                        print(f"Fracción: {fraccion}")
                        print(f"Épocas: {max_epochs}")
                        print(str(e))

                        if "out of memory" in str(e).lower():
                            print("Tu PC no aguantó esta configuración por memoria.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        continue

    df = pd.DataFrame(resultados)

    if len(df) > 0:
        df_ordenado = df.sort_values(
            by=["test_mae", "test_rmse"],
            ascending=True,
        )

        df_ordenado.to_csv(
            Path(args.output_dir) / "ranking_modelos.csv",
            index=False,
        )

        print("\nMEJORES RESULTADOS")
        print(df_ordenado.head(20))

        plt.figure(figsize=(12, 6))

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

        ruta_comparacion = Path(args.output_dir) / "comparacion_modelos_mae.png"
        plt.savefig(ruta_comparacion, dpi=300)
        plt.show()

        print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()