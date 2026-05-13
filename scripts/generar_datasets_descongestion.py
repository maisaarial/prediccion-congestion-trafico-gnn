from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath("src"))

import pandas as pd
import torch
import yaml

from traffic_gnn.data.io import load_traffic_files, read_sensors_csv
from traffic_gnn.features.congestion import calcular_congestion
from traffic_gnn.features.temporal import obtener_variables_temporales
from traffic_gnn.clustering.proximity import generar_cluster_proximidad
from traffic_gnn.clustering.behavior import (
    calcular_pivote_cl_comp,
    generar_cluster_comportamiento,
)
from traffic_gnn.clustering.intersections import (
    intersectar_clusters,
    intersectar_clusters_sentido_v1,
    intersectar_clusters_sentido_v2,
)
from traffic_gnn.clustering.direction import calcular_sentido_mejorado
from traffic_gnn.graph.aggregation import (
    aggregation_congestion_por_clusters,
    calcular_centroides_clusters,
    reindex_clusters_dataframe,
)
from traffic_gnn.graph.adjacency import (
    adjacency_knn_from_centroids,
    adjacency_correlation_topk,
    build_cluster_time_matrix,
    reindex_time_matrix,
    to_pyg_tensors,
)
from traffic_gnn.graph.datasets import (
    crear_ventanas,
    split_temporal,
    build_tensor_datasets,
)


CASOS_DEFAULT = [
    "sensores_tal_cual",
    "proximidad",
    "proximidad_comportamiento",
    "proximidad_sentido_v1",
    "proximidad_sentido_v2",
]

ADY_DEFAULT = ["cercania", "correlacion"]

CASOS_CON_CLUSTER = {
    "proximidad",
    "proximidad_comportamiento",
    "proximidad_sentido_v1",
    "proximidad_sentido_v2",
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def validar_columnas(df: pd.DataFrame, required: list[str], nombre: str) -> None:
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas en {nombre}: {faltantes}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def cargar_datos(
    cfg: dict,
    months: list[str] | None,
    all_months: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cfg["paths"]
    cols_cfg = cfg["columns"]
    data_cfg = cfg.get("data", {})

    sep = data_cfg.get("sep", ";")
    encoding = data_cfg.get("encoding", "latin-1")

    months = months or data_cfg.get("months")
    use_all = bool(all_months or data_cfg.get("use_all_months", False))

    cols_trafico = list(cols_cfg["traffic"].values())

    trafico = load_traffic_files(
        paths["traffic_dir"],
        months=months,
        use_all_months=use_all,
        usecols=cols_trafico,
        sep=sep,
        encoding=encoding,
    )
    trafico = normalize_columns(trafico)

    cols_sensores = list(cols_cfg["sensors"].values())
    sensores_path = Path(paths["sensors_csv"])

    if not sensores_path.exists():
        sensors_dir = Path(paths.get("sensors_dir", "data/raw/sensores"))
        csvs = sorted(sensors_dir.rglob("*.csv"))

        if not csvs:
            raise FileNotFoundError(
                f"No encuentro sensores_csv={sensores_path}. "
                "Guarda el archivo de sensores en data/raw/sensores/."
            )

        sensores_path = csvs[0]
        print(f"Usando archivo de sensores encontrado automáticamente: {sensores_path}")

    sensores = read_sensors_csv(
        sensores_path,
        usecols=cols_sensores,
        sep=sep,
        encoding=encoding,
    )
    sensores = normalize_columns(sensores)

    return trafico, sensores


def preparar_base(
    trafico: pd.DataFrame,
    sensores: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tc = cfg["columns"]["traffic"]
    sc = cfg["columns"]["sensors"]

    validar_columnas(
        trafico,
        [tc["id"], tc["fecha"], tc["intensidad"], tc["ocupacion"]],
        "tráfico",
    )

    validar_columnas(
        sensores,
        [sc["id"], sc["utm_x"], sc["utm_y"], sc["longitud"], sc["latitud"]],
        "sensores",
    )

    trafico = trafico.rename(
        columns={
            tc["id"]: "id",
            tc["fecha"]: "fecha",
            tc["intensidad"]: "intensidad",
            tc["ocupacion"]: "ocupacion",
        }
    )

    sensores = sensores.rename(
        columns={
            sc["id"]: "id",
            sc["utm_x"]: "utm_x",
            sc["utm_y"]: "utm_y",
            sc["longitud"]: "longitud",
            sc["latitud"]: "latitud",
        }
    )

    trafico = calcular_congestion(trafico)
    trafico = obtener_variables_temporales(trafico)

    if trafico["congestion"].max() <= 1:
        trafico["descongestion"] = 1 - trafico["congestion"]
    else:
        trafico["descongestion"] = 100 - trafico["congestion"]

    trafico["congestion"] = trafico["descongestion"]

    sensores["id"] = pd.to_numeric(sensores["id"], errors="coerce").astype("Int64")
    trafico["id"] = pd.to_numeric(trafico["id"], errors="coerce").astype("Int64")

    sensores = sensores.dropna(subset=["id", "utm_x", "utm_y"]).copy()
    trafico = trafico.dropna(subset=["id", "fecha"]).copy()

    sensores["id"] = sensores["id"].astype(int)
    trafico["id"] = trafico["id"].astype(int)

    return trafico, sensores


def contar_clusters(df: pd.DataFrame, cluster_col: str) -> int:
    vals = df[cluster_col].dropna().unique()
    vals = [v for v in vals if int(v) != -1]
    return len(vals)


def generar_cluster_proximidad_target(
    sensores: pd.DataFrame,
    cfg: dict,
    target: int,
    cluster_col: str = "cluster_proximidad",
) -> pd.DataFrame:
    cl_cfg = cfg["clustering"]
    prox_cfg = cl_cfg["proximity"]

    min_samples = int(prox_cfg.get("min_samples", 1))

    print(
        f"Buscando epsilon automáticamente para target={target}..."
    )

    eps_min = float(prox_cfg.get("eps_search_min", 10))
    eps_max = float(prox_cfg.get("eps_search_max", 1000))
    n_iter = int(prox_cfg.get("eps_search_iter", 25))

    best_df = None
    best_eps = None
    best_diff = float("inf")
    best_n = None

    lo = eps_min
    hi = eps_max

    for _ in range(n_iter):
        mid = (lo + hi) / 2

        df = generar_cluster_proximidad(
            sensores,
            epsilon=mid,
            min_samples=min_samples,
            cluster_col=cluster_col,
        )

        n_clusters = contar_clusters(df, cluster_col)
        diff = abs(n_clusters - target)

        if diff < best_diff:
            best_df = df
            best_eps = mid
            best_diff = diff
            best_n = n_clusters

        if n_clusters > target:
            lo = mid
        else:
            hi = mid

    print(
        f"Cluster proximidad target={target}: "
        f"eps_aprox={best_eps:.4f}, "
        f"clusters_obtenidos={best_n}, "
        f"diferencia={best_diff}"
    )

    if best_df is None:
        raise RuntimeError(
            f"No se pudo construir cluster de proximidad para target={target}"
        )

    return best_df

    print(
        f"No existe {eps_key} en params.yaml. "
        f"Buscando epsilon automáticamente para target={target}..."
    )

    eps_min = float(prox_cfg.get("eps_search_min", 10))
    eps_max = float(prox_cfg.get("eps_search_max", 1000))
    n_iter = int(prox_cfg.get("eps_search_iter", 25))

    best_df = None
    best_eps = None
    best_diff = float("inf")
    best_n = None

    lo = eps_min
    hi = eps_max

    for _ in range(n_iter):
        mid = (lo + hi) / 2

        df = generar_cluster_proximidad(
            sensores,
            epsilon=mid,
            min_samples=min_samples,
            cluster_col=cluster_col,
        )

        n_clusters = contar_clusters(df, cluster_col)
        diff = abs(n_clusters - target)

        if diff < best_diff:
            best_df = df
            best_eps = mid
            best_diff = diff
            best_n = n_clusters

        if n_clusters > target:
            lo = mid
        else:
            hi = mid

    print(
        f"Cluster proximidad target={target}: eps_aprox={best_eps:.4f}, "
        f"clusters_obtenidos={best_n}, diferencia={best_diff}"
    )

    if best_df is None:
        raise RuntimeError(f"No se pudo construir cluster de proximidad para target={target}")

    return best_df


def expandir_casos_solicitados(
    casos: list[str],
    cluster_targets: list[int],
) -> list[str]:
    casos_expandidos: list[str] = []

    for caso in casos:
        if caso == "sensores_tal_cual":
            casos_expandidos.append(caso)

        elif caso in CASOS_CON_CLUSTER:
            for target in cluster_targets:
                casos_expandidos.append(f"{caso}_{target}")

        else:
            casos_expandidos.append(caso)

    return casos_expandidos


def construir_casos(
    trafico: pd.DataFrame,
    sensores: pd.DataFrame,
    cfg: dict,
    incluir_sentido: bool = True,
    graphml_path=None,
    cluster_targets=None,
) -> dict[str, tuple[pd.DataFrame, str]]:
    if cluster_targets is None:
        cluster_targets = [50, 500]

    cl_cfg = cfg["clustering"]

    casos: dict[str, tuple[pd.DataFrame, str]] = {}

    sensores_base = sensores.copy()
    sensores_base["cluster_sensor"] = pd.factorize(sensores_base["id"])[0]
    casos["sensores_tal_cual"] = (sensores_base, "cluster_sensor")

    pivote = calcular_pivote_cl_comp(trafico)

    cluster_comp = generar_cluster_comportamiento(
        pivote,
        n_clusters=int(cl_cfg["behavior"]["n_clusters"]),
        random_state=int(cl_cfg["behavior"].get("random_state", 42)),
        column_name="cluster_comportamiento",
    ).reset_index()[["id", "cluster_comportamiento"]]

    if graphml_path is None:
        graphml_path = cfg["paths"].get("graphml_path", "data/raw/osm/madrid_drive.graphml")

    graphml_path = Path(graphml_path)

    sentido = None

    if incluir_sentido and graphml_path.exists():
        try:
            print(f"Calculando sentido con graphml: {graphml_path}")
            sentido = calcular_sentido_mejorado(sensores, graphml_path=graphml_path)

        except Exception as exc:
            print(f"No se pudo calcular sentido. Se omiten casos de sentido. Motivo: {exc}")
            sentido = None

    elif incluir_sentido:
        print("No se construyen casos de sentido porque no existe graphml_path.")

    else:
        print("No se construyen casos de sentido porque se usó --skip_sentido.")

    for target in cluster_targets:
        print("\n" + "=" * 80)
        print(f"Construyendo casos con objetivo de clusters: {target}")
        print("=" * 80)

        cluster_col_prox = f"cluster_proximidad_{target}"

        cluster_proximidad = generar_cluster_proximidad_target(
            sensores=sensores,
            cfg=cfg,
            target=int(target),
            cluster_col=cluster_col_prox,
        )

        casos[f"proximidad_{target}"] = (
            cluster_proximidad,
            cluster_col_prox,
        )

        prox_comp_col = f"cluster_prox_comportamiento_{target}"

        prox_comp = intersectar_clusters(
            cluster_proximidad,
            cluster_comp,
            prox_col=cluster_col_prox,
            behavior_col="cluster_comportamiento",
            output_col=prox_comp_col,
        )

        casos[f"proximidad_comportamiento_{target}"] = (
            prox_comp,
            prox_comp_col,
        )

        if sentido is not None:
            try:
                prox_sent_v1_col = f"cluster_prox_sentido_v1_{target}"
                prox_sent_v2_col = f"cluster_prox_sentido_v2_{target}"

                prox_sent_v1 = intersectar_clusters_sentido_v1(
                    cluster_proximidad,
                    sentido,
                    prox_col=cluster_col_prox,
                    output_col=prox_sent_v1_col,
                )

                prox_sent_v2 = intersectar_clusters_sentido_v2(
                    cluster_proximidad,
                    sentido,
                    prox_col=cluster_col_prox,
                    output_col=prox_sent_v2_col,
                )

                casos[f"proximidad_sentido_v1_{target}"] = (
                    prox_sent_v1,
                    prox_sent_v1_col,
                )

                casos[f"proximidad_sentido_v2_{target}"] = (
                    prox_sent_v2,
                    prox_sent_v2_col,
                )

            except Exception as exc:
                print(
                    f"No se pudieron construir casos de sentido para target={target}. "
                    f"Se omiten. Motivo: {exc}"
                )

    return casos


def construir_y_guardar_dataset(
    nombre_caso: str,
    df_cluster: pd.DataFrame,
    cluster_col: str,
    tipo_adyacencia: str,
    trafico: pd.DataFrame,
    cfg: dict,
    output_dir: str | Path,
) -> None:
    graph_cfg = cfg["graph"]
    tr_cfg = cfg["training"]

    df_cluster_re, _, mapping_inv = reindex_clusters_dataframe(df_cluster, cluster_col)

    trafico_cluster = trafico.merge(
        df_cluster_re[["id", cluster_col, "utm_x", "utm_y"]],
        on="id",
        how="inner",
    )

    if trafico_cluster.empty:
        raise ValueError(f"No hay registros tras unir tráfico con clusters para {nombre_caso}.")

    agg = aggregation_congestion_por_clusters(
        trafico_cluster,
        nombre_col_cluster=cluster_col,
    )

    tabla_original = build_cluster_time_matrix(agg)

    if tipo_adyacencia == "correlacion":
        tabla_gnn, edges_df, mapping, mapping_inverso = adjacency_correlation_topk(
            tabla_original,
            k=int(graph_cfg.get("top_k_correlations", 5)),
        )

    elif tipo_adyacencia == "cercania":
        tabla_gnn, mapping, mapping_inverso = reindex_time_matrix(tabla_original)

        centroides_orig = calcular_centroides_clusters(
            df_cluster_re,
            cluster_col=cluster_col,
        )

        centroides_orig = centroides_orig[
            centroides_orig["cluster"].isin(mapping.keys())
        ].copy()

        centroides_orig["cluster"] = centroides_orig["cluster"].map(mapping).astype(int)
        centroides_orig = centroides_orig.sort_values("cluster")

        edges_df = adjacency_knn_from_centroids(
            centroides_orig,
            k=int(graph_cfg.get("top_k_neighbors", 5)),
        )

    else:
        raise ValueError(f"tipo_adyacencia no válido: {tipo_adyacencia}")

    edge_index, edge_weight = to_pyg_tensors(edges_df)

    num_nodes = tabla_gnn.shape[1]
    max_edge = int(edge_index.max().item()) if edge_index.numel() else -1

    if max_edge >= num_nodes:
        raise ValueError(
            f"edge_index inválido para {nombre_caso}/{tipo_adyacencia}: "
            f"max_edge={max_edge}, num_nodes={num_nodes}"
        )

    data = tabla_gnn.values.astype("float32")

    X, y = crear_ventanas(
        data,
        window=int(tr_cfg.get("window", 12)),
        horizon=int(tr_cfg.get("horizon", 1)),
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_temporal(
        X,
        y,
        train_ratio=float(tr_cfg.get("train_ratio", 0.70)),
        val_ratio=float(tr_cfg.get("val_ratio", 0.15)),
    )

    train_dataset, val_dataset, test_dataset = build_tensor_datasets(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    out_dir = Path(output_dir) / nombre_caso / tipo_adyacencia
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_dataset, out_dir / "train.pt")
    torch.save(val_dataset, out_dir / "val.pt")
    torch.save(test_dataset, out_dir / "test.pt")

    torch.save(
        {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_nodes": num_nodes,
            "num_features": X_train.shape[3],
            "seq_len": X_train.shape[1],
            "tabla_columns_mapping": mapping,
            "tabla_columns_mapping_inverso": mapping_inverso,
            "cluster_mapping_inverso": mapping_inv,
            "caso": nombre_caso,
            "tipo_adyacencia": tipo_adyacencia,
        },
        out_dir / "graph.pt",
    )

    tabla_gnn.to_csv(out_dir / "tabla_gnn.csv")
    edges_df.to_csv(out_dir / "edges.csv", index=False)

    print(
        f"Guardado {nombre_caso}/{tipo_adyacencia}: "
        f"X_train={tuple(X_train.shape)}, "
        f"y_train={tuple(y_train.shape)}, "
        f"N={num_nodes}, "
        f"E={edge_index.shape[1]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Genera datasets GNN locales por caso y tipo de adyacencia."
    )

    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Meses a cargar, por ejemplo: 01-2025 02-2025",
    )
    parser.add_argument(
        "--all_months",
        action="store_true",
        help="Usar todos los CSV en data/raw/trafico",
    )
    parser.add_argument("--casos", nargs="+", default=CASOS_DEFAULT)
    parser.add_argument("--adyacencias", nargs="+", default=ADY_DEFAULT)
    parser.add_argument(
        "--skip_sentido",
        action="store_true",
        help="Omitir casos de proximidad_sentido_v1/v2",
    )
    parser.add_argument(
        "--graphml_path",
        default="data/raw/osm/madrid_drive.graphml",
    )
    parser.add_argument(
        "--cluster_targets",
        nargs="+",
        type=int,
        default=[50, 500],
        help="Número objetivo de clusters para los casos agregados",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["paths"]["graphml_path"] = args.graphml_path

    trafico, sensores = cargar_datos(
        cfg,
        months=args.months,
        all_months=args.all_months,
    )

    trafico, sensores = preparar_base(trafico, sensores, cfg)

    casos = construir_casos(
        trafico,
        sensores,
        cfg,
        incluir_sentido=not args.skip_sentido,
        graphml_path=args.graphml_path,
        cluster_targets=args.cluster_targets,
    )

    casos_solicitados = expandir_casos_solicitados(
        casos=args.casos,
        cluster_targets=args.cluster_targets,
    )

    processed_dir = "data/processed_descongestion"

    for nombre_caso in casos_solicitados:
        if nombre_caso not in casos:
            print(f"Caso no disponible y se omite: {nombre_caso}")
            continue

        df_cluster, cluster_col = casos[nombre_caso]

        for tipo_ady in args.adyacencias:
            try:
                construir_y_guardar_dataset(
                    nombre_caso,
                    df_cluster,
                    cluster_col,
                    tipo_ady,
                    trafico,
                    cfg,
                    processed_dir,
                )

            except Exception as exc:
                print(f"ERROR generando {nombre_caso}/{tipo_ady}: {exc}")


if __name__ == "__main__":
    main()