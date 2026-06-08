# Realtime App MVP

MVP/demo de visualizacion en tiempo real para trafico de Madrid. Esta carpeta no modifica el pipeline original del TFM ni ejecuta entrenamiento.

## Modo demo y modelo temporal

La aplicacion descarga `https://informo.madrid.es/informo/tmadrid/pm.xml`, guarda la captura XML, parsea los puntos de medida y crea salidas CSV dentro de `realtime_app/data/`.

El sistema puede trabajar en dos modos:

- `REAL_MODEL_TEMPORAL`: usa el checkpoint temporal copiado en `artifacts/selected_model/`.
- `DEMO_HISTORICO_INSUFICIENTE`: usa fallback mock cuando todavia no hay 12 bloques de 15 minutos disponibles.

El fallback mock mantiene:

```text
prediccion_15min = congestion_actual
diferencia = prediccion_15min - congestion_actual
```

## Calculo de congestion

Para cada sensor del XML se calcula una senal provisional:

```text
congestion_demo = ocupacion / 100
```

El valor se recorta al rango `[0, 1]`.

Cuando existe mapeo sensor -> nodo de `proximidad_500/correlacion`, la congestion del nodo se calcula como:

```text
congestion_actual = promedio(congestion_demo de los sensores activos del nodo)
```

Si el mapeo no esta disponible, la app cae a una capa provisional de puntos de medida del XML.

## Inferencia real temporal

El checkpoint temporal requiere una ventana de 12 bloques de 15 minutos. La entrada se construye con forma:

```text
[batch=1, window=12, num_nodes=481, features=1]
```

La feature usada es `congestion`, tomada de `congestion_actual` por nodo y ordenada segun `tabla_gnn.csv` / `graph.pt`.

Si no hay 12 bloques historicos distintos en `realtime_app/data/blocks_15min/`, no se ejecuta el modelo real y la API devuelve `DEMO_HISTORICO_INSUFICIENTE`.

## Recolector automatico

FastAPI arranca un recolector automatico al iniciar el backend. El recolector descarga el XML cada 5 minutos, parsea los datos, evita duplicados por `fecha_hora` y guarda nuevos snapshots en:

- `data/captures/xml/`
- `data/captures/parsed/`
- `data/blocks_15min/`
- `data/predictions/`

La ventana del modelo requiere 12 bloques de 15 minutos, es decir, 3 horas de historico. El endpoint `GET /history/status` informa si la ventana ya esta lista.

Este checkpoint es solo de integracion tecnica. El modelo final debera reemplazar:

- `artifacts/selected_model/model.pt`
- `artifacts/selected_model/model_config.json`

No se deben mezclar modelos y grafos de configuraciones distintas. Si se cambia `model.pt`, tambien deben corresponder `graph.pt`, `edges.csv`, `tabla_gnn.csv` y `model_config.json`.

## Artefactos futuros

El slot `artifacts/selected_model/` esta preparado para copiar mas adelante:

- `model.pt`
- `graph.pt`
- `edges.csv`
- `tabla_gnn.csv`
- `model_config.json`

Cuando exista el checkpoint real, se puede reemplazar el servicio mock por un servicio de inferencia que mantenga el contrato de salida de la API.
