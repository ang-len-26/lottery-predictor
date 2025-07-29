from src.data_preprocessing import load_data, build_features_targets, build_single_feature_vector
from src.model_training import train_model, load_trained_model
from src.model_prediction import predict_next_draw
from src.evaluation import jaccard_similarity
import pandas as pd
import numpy as np
import os

LOG_PATH = "logs/prediction_log.csv"
DATA_PATH = "data/historial_sorteos.xlsx"

os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    print("🔁 Cargando historial y modelo...")
    df = load_data(DATA_PATH)
    model = load_trained_model()
    print("✅ Modelo cargado.\n")

    # PREDICCIÓN ACTUAL
    print("🔮 Prediciendo el próximo sorteo (no registrado)...")
    future_input = build_single_feature_vector(df)
    future_prediction = predict_next_draw(model, future_input)
    print("🎯 Predicción para el siguiente sorteo:", future_prediction)

    # NUEVO SORTEO MANUAL
    fecha = input("\n📅 Ingresa la fecha del nuevo sorteo (formato YYYY-MM-DD): ")
    resultado_real_str = input("📝 Ingresa los 6 números ganadores del sorteo (separados por coma): ")
    resultado_real = [int(x.strip()) for x in resultado_real_str.split(",")]

    # VECTOR BINARIO DE RESULTADO REAL
    real_vector = np.zeros(50, dtype=int)
    for n in resultado_real:
        if 1 <= n <= 50:
            real_vector[n - 1] = 1

    # EVALUAR PRECISIÓN
    predicted_vector = np.zeros(50, dtype=int)
    for n in future_prediction:
        predicted_vector[n - 1] = 1

    jaccard = jaccard_similarity(real_vector, predicted_vector)
    print(f"\n📊 Precisión de la predicción anterior (Jaccard): {jaccard:.2f}")

    # GUARDAR EN LOG
    log_entry = pd.DataFrame([{
        "Fecha": fecha,
        "Predicción": future_prediction,
        "Resultado Real": resultado_real,
        "Aciertos": sum(real_vector & predicted_vector),
        "Jaccard": round(jaccard, 2)
    }])
    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)
    print("📁 Predicción y resultados guardados en", LOG_PATH)

    # AGREGAR AL HISTORIAL
    new_row = {"Fecha": pd.to_datetime(fecha)}
    for i, n in enumerate(resultado_real):
        new_row[f"N{i+1}"] = n
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(by="Fecha")
    df.to_excel(DATA_PATH, index=False)
    print("📁 Nuevo sorteo agregado al historial.")

    # REENTRENAMIENTO
    print("\n♻️ Reentrenando modelo con historial actualizado...")
    X, y = build_features_targets(df)
    train_model(X, y)
    print("\n✅ Retroalimentación completa. ¡Listo para predecir el siguiente sorteo!")
