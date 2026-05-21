import pandas as pd
from pathlib import Path

# Archivo original
archivo = Path(r"results\experimentos_epocas_datos\run_20260518_131858\resultados_completos.csv")

# Leer CSV
df = pd.read_csv(archivo)

# Columnas que quieres conservar
columnas = [
    "caso",
    "tipo_adyacencia",
    "modelo",
    "best_epoch",
    "best_val_loss",
    "test_loss",
    "test_mae",
    "test_rmse",
    "tiempo_segundos"
]

df = df[columnas]

# Calcular tiempo en minutos y horas
df["tiempo_minutos"] = (df["tiempo_segundos"] / 60).round(2)
df["tiempo_horas"] = (df["tiempo_segundos"] / 3600).round(2)

# Ordenar por mejores métricas
df = df.sort_values(
    by=["test_mae", "test_rmse"],
    ascending=[True, True]
)

# Guardar en la misma carpeta
salida = archivo.parent / "ranking_modelos_resumen.csv"
df.to_csv(salida, index=False)

print("Archivo generado correctamente:")
print(salida)