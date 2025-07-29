from src.data_preprocessing import load_data, build_features_targets, build_single_feature_vector
from src.model_training import train_model
from src.model_prediction import predict_next_draw
from src.evaluation import jaccard_similarity
import numpy as np

if __name__ == "__main__":
    print("📊 Cargando datos...")
    df = load_data("data/historial_sorteos.xlsx")
    print(f"✔️  Sorteos cargados: {len(df)}")

    print("🧠 Preparando datos de entrenamiento...")
    X, y = build_features_targets(df)
    print(f"✔️  Datos de entrenamiento preparados: {X.shape[0]} muestras")

    print("🔧 Entrenando modelo...")
    model = train_model(X, y)

    print("Última predicción (basada en últimos 10 sorteos):")
    last_prediction = predict_next_draw(model, X[-1])
    print("→ Números más probables:", last_prediction)

    jaccard = jaccard_similarity(y[-1], model.predict(np.array([X[-1]]))[0])
    print(f"Jaccard similarity con el último sorteo real: {jaccard:.2f}")

    print("📈 Predicción para el siguiente sorteo (NO registrado):")
    future_input = build_single_feature_vector(df)
    future_prediction = predict_next_draw(model, future_input)
    print("→ Números más probables (futuros):", future_prediction)
