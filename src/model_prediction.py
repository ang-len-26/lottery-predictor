import numpy as np

def predict_next_draw(model, input_vector, top_n=6):
    probabilities = model.predict(np.array([input_vector]), verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return (top_indices + 1).tolist()
