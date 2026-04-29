# Predicción de congestión vehicular con GNN

Pipeline local para construir grafos de tráfico de Madrid y comparar arquitecturas espaciotemporales:

- GCN + LSTM
- GCN + GRU
- GAT + LSTM

La estructura experimental correcta es:

```text
caso de nodos × tipo de adyacencia × modelo
```

Casos de nodos:

- `sensores_tal_cual`
- `proximidad`
- `proximidad_comportamiento`
- `proximidad_sentido_v1`
- `proximidad_sentido_v2`

Tipos de matriz de adyacencia:

- `cercania`
- `correlacion`

## Instalación

```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Dónde guardar los datos

Tráfico:

```text
data/raw/trafico/01-2025.csv
data/raw/trafico/02-2025.csv
...
```

Sensores:

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

## Prueba rápida local

```bash
python scripts/run_pipeline_completo.py --months 01-2025 --skip_sentido --casos proximidad proximidad_comportamiento --fracciones 0.25 --epochs 5 --batch_size 8
```

## Generar datasets solamente

```bash
python scripts/generar_datasets.py --months 01-2025 --skip_sentido --casos proximidad proximidad_comportamiento --adyacencias cercania correlacion
```

## Entrenar modelos solamente

```bash
python scripts/probar_epocas_datos.py --casos proximidad proximidad_comportamiento --adyacencias cercania correlacion --fracciones 0.25 --epochs 5 --batch_size 8
```

## Usar varios meses

```bash
python scripts/run_pipeline_completo.py --months 01-2025 02-2025 03-2025 --skip_sentido --fracciones 0.25 --epochs 5 --batch_size 8
```

## Usar todos los CSV

```bash
python scripts/run_pipeline_completo.py --all_months --skip_sentido --fracciones 0.25 --epochs 5 --batch_size 8
```

## Resultados

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

La carpeta `pred_vs_real` contiene gráficas comparando la congestión real contra la predicha para nodos representativos.

Más detalles en:

```text
docs/INSTRUCCIONES_PIPELINE_LOCAL.md
```
