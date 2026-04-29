# Pipeline local para experimentos GNN de congestión

Este proyecto queda preparado para trabajar sin Colab. El flujo completo es:

1. Leer CSV locales de tráfico y sensores.
2. Construir casos de nodos.
3. Construir dos matrices de adyacencia por caso: cercanía y correlación.
4. Generar `train.pt`, `val.pt`, `test.pt` y `graph.pt`.
5. Entrenar `GCN_LSTM`, `GCN_GRU` y `GAT_LSTM`.
6. Guardar métricas, modelos, curvas de entrenamiento y gráficas real vs predicción.

## 1. Dónde guardar los CSV

Guarda los CSV de tráfico aquí:

```text
data/raw/trafico/
├── 01-2025.csv
├── 02-2025.csv
└── ...
```

Guarda el CSV de ubicación de sensores aquí:

```text
data/raw/sensores/pmed_ubicacion_01-2025.csv
```

Columnas esperadas de tráfico:

```text
id, fecha, tipo_elem, intensidad, ocupacion
```

Columnas esperadas de sensores:

```text
id, distrito, nombre, utm_x, utm_y, longitud, latitud
```

Si el archivo de sensores tiene otro nombre, actualiza `configs/params.yaml` en:

```yaml
paths:
  sensors_csv: data/raw/sensores/TU_ARCHIVO.csv
```

## 2. Ejecutar prueba pequeña

Para probar solo enero 2025, sin casos de sentido:

```bash
python scripts/run_pipeline_completo.py --months 01-2025 --skip_sentido --casos proximidad proximidad_comportamiento --fracciones 0.25 --epochs 5 --batch_size 8
```

## 3. Ejecutar solo generación de datasets

```bash
python scripts/generar_datasets.py --months 01-2025 --skip_sentido --casos proximidad proximidad_comportamiento --adyacencias cercania correlacion
```

## 4. Ejecutar solo entrenamiento

```bash
python scripts/probar_epocas_datos.py --casos proximidad proximidad_comportamiento --adyacencias cercania correlacion --fracciones 0.25 --epochs 5 --batch_size 8
```

## 5. Usar varios meses

```bash
python scripts/run_pipeline_completo.py --months 01-2025 02-2025 03-2025 --skip_sentido --casos proximidad proximidad_comportamiento --fracciones 0.25 --epochs 5 --batch_size 8
```

## 6. Usar todos los CSV disponibles

```bash
python scripts/run_pipeline_completo.py --all_months --skip_sentido --fracciones 0.25 --epochs 5 --batch_size 8
```

## 7. Casos de sentido

Los casos `proximidad_sentido_v1` y `proximidad_sentido_v2` necesitan el archivo GraphML de OSM:

```text
data/external/madrid_drive.graphml
```

Si no lo tienes, usa `--skip_sentido`.

## 8. Resultados generados

```text
results/experimentos_epocas_datos/
├── resultados_completos.csv
├── ranking_modelos.csv
├── modelos/
├── historiales/
└── graficas/
    ├── curvas_entrenamiento/
    ├── pred_vs_real/
    └── comparaciones/
```

Las gráficas `pred_vs_real` comparan la serie real con la predicción para nodos representativos.

## 9. Estructura conceptual correcta

```text
CASO DE NODOS
+
TIPO DE ADYACENCIA
+
MODELO
```

Casos de nodos:

```text
sensores_tal_cual
proximidad
proximidad_comportamiento
proximidad_sentido_v1
proximidad_sentido_v2
```

Tipos de adyacencia:

```text
cercania
correlacion
```

Modelos:

```text
GCN_LSTM
GCN_GRU
GAT_LSTM
```
