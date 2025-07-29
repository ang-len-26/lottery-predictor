import numpy as np

def jaccard_similarity(y_true, y_pred):
    true_set = set(np.where(y_true == 1)[0])
    pred_set = set(np.where(y_pred >= 0.5)[0])
    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)
    return intersection / union if union != 0 else 0.0
