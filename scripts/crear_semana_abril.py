import pandas as pd
from pathlib import Path

# Archivo original
entrada = Path("data/raw/trafico/04-2025.csv")

# Archivo de salida
salida = Path("data/raw/trafico/04-2025_semana.csv")

print("Leyendo archivo...")
df = pd.read_csv(entrada, sep=";")

# Convertir fecha
df["fecha"] = pd.to_datetime(df["fecha"])

# Semana seleccionada
inicio = "2025-04-07"
fin = "2025-04-13 23:59:59"

# Filtrar
df_semana = df[
    (df["fecha"] >= inicio) &
    (df["fecha"] <= fin)
]

# Guardar
df_semana.to_csv(salida, sep=";", index=False)

print("\nArchivo creado:")
print(salida)

print("\nResumen:")
print("Filas originales:", len(df))
print("Filas semana:", len(df_semana))
print("Desde:", df_semana["fecha"].min())
print("Hasta:", df_semana["fecha"].max())