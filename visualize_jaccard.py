import pandas as pd
import matplotlib.pyplot as plt

# Cargar el log
log_path = "logs/prediction_log.csv"
df = pd.read_csv(log_path)

# Asegurarse que la columna 'Fecha' sea tipo datetime
df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
df = df.dropna(subset=["Fecha", "Jaccard"])  # limpiar errores

# Ordenar por fecha por si acaso
df = df.sort_values("Fecha")

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Jaccard"], marker='o', linestyle='-', color='blue', label="Jaccard")

# Opcional: Promedio móvil para suavizar
df["Jaccard SMA(5)"] = df["Jaccard"].rolling(window=5).mean()
plt.plot(df["Fecha"], df["Jaccard SMA(5)"], color='orange', linestyle='--', label="Promedio móvil (5)")

# Decoración
plt.title("📈 Evolución del Puntaje Jaccard por Sorteo")
plt.xlabel("Fecha del sorteo")
plt.ylabel("Jaccard similarity")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Mostrar
plt.show()
