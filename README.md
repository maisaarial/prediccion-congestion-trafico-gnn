# TFM - Predicción de congestión vehicular con grafos y GNN

Repositorio base para organizar de forma reproducible el proyecto del TFM sobre predicción de congestión vehicular en Madrid a partir de sensores de tráfico, construcción de grafos y modelos GNN.

## Objetivo

Este repositorio deja estructurado el flujo completo del proyecto para trabajar con GitHub + Colab Pro+:

1. Cargar datos de tráfico y puntos de medida.
2. Calcular la variable de congestión.
3. Construir los 3 casos de nodos:
   - `proximidad`
   - `proximidad_comportamiento`
   - `proximidad_sentido`
4. Generar versiones con `50` y `500` clusters.
5. Construir matrices de adyacencia por:
   - `cercania`
   - `correlacion`
6. Preparar tensores para GNN.
7. Entrenar un baseline `GCN + LSTM`.
8. Entrenar un baseline `GCN+GRU`
9. Entrenar un baseline `GAT+LSTM`

## Estructura del proyecto

```text
traffic_gnn_madrid/
├── configs/
│   └── params.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   └── 01_colab_github_pipeline.ipynb
├── results/
│   ├── adjacency/
│   ├── clusters/
│   ├── figures/
│   └── models/
├── src/
│   └── traffic_gnn/
│       ├── data/
│       │   └── io.py
│       ├── features/
│       │   ├── congestion.py
│       │   └── temporal.py
│       ├── clustering/
│       │   ├── behavior.py
│       │   ├── direction.py
│       │   ├── intersections.py
│       │   └── proximity.py
│       ├── graph/
│       │   ├── adjacency.py
│       │   ├── aggregation.py
│       │   └── datasets.py
│       ├── models/
│       │   └── gcn_lstm.py
│       ├── training/
│       │   └── engine.py
│       └── pipelines/
│           ├── build_adjacency.py
│           └── build_clusters.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Casos contemplados

### 1. Proximidad
Clustering espacial directo sobre coordenadas UTM de los sensores.

### 2. Proximidad ∩ Comportamiento
Intersección entre:
- cluster espacial por proximidad
- cluster de comportamiento temporal a partir de la serie agregada por franjas horarias

### 3. Proximidad ∩ Sentido
Intersección entre:
- cluster espacial por proximidad
- agrupación por sentido de circulación estimado a partir de OSM / bearings

## Qué hace cada módulo

### `features/congestion.py`
Calcula la congestión a partir de intensidad y ocupación.

### `features/temporal.py`
Construye variables temporales como:
- fecha en datetime
- día de semana
- tipo de día (`L` / `F`)
- franja horaria de 2 horas
- etiqueta temporal para clustering de comportamiento

### `clustering/proximity.py`
Genera clusters espaciales con DBSCAN sobre `utm_x`, `utm_y`.

### `clustering/behavior.py`
Genera la tabla pivote de comportamiento y aplica KMeans.

### `clustering/direction.py`
Carga un grafo de OSM y estima el sentido dominante de cada sensor usando la arista más cercana.

### `clustering/intersections.py`
Interseca clusters de distintos criterios y reasigna etiquetas consecutivas.

### `graph/aggregation.py`
Agrega la congestión por cluster y calcula centroides.

### `graph/adjacency.py`
Construye grafos por:
- k vecinos más cercanos
- top-k correlaciones

### `graph/datasets.py`
Construye ventanas temporales, split temporal y tensores para entrenamiento.

### `models/gcn_lstm.py`
Modelo baseline GCN + LSTM en PyTorch Geometric.

### `training/engine.py`
Entrenamiento y evaluación.

## Cómo usarlo en Colab Pro+

## Flujo sugerido de trabajo

### Paso 1. Construcción de clusters
Se usa `build_clusters.py` o el notebook para generar:
- `cl_proximidad_50.csv`
- `cl_proximidad_500.csv`
- `cl_proximidad_comportamiento_50.csv`
- `cl_proximidad_comportamiento_500.csv`
- `cl_proximidad_sentido_50.csv`
- `cl_proximidad_sentido_500.csv`

### Paso 2. Construcción de adyacencias
Para cada caso y tamaño:
- adyacencia por cercanía
- adyacencia por correlación

### Paso 3. Preparación GNN
- agregación por cluster
- tabla temporal por nodo
- ventanas temporales
- `edge_index` y `edge_weight`

### Paso 4. Entrenamiento
Entrenar baselines `GCN + LSTM`, `GCN+GRU` y `GAT+LSTM`.
