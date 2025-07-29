import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

MODEL_PATH = "models/lottery_model.h5"

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(50, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y):
    model = build_model(X.shape[1])
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print("✅ Modelo entrenado y guardado.")
    return model

def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)
