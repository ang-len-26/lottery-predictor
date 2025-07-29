import pandas as pd
import numpy as np

def get_max_number_for_date(date):
    if date <= pd.Timestamp("1997-08-31"):
        return 36
    elif date <= pd.Timestamp("1998-02-28"):
        return 40
    elif date <= pd.Timestamp("2000-06-30"):
        return 43
    elif date <= pd.Timestamp("2022-01-31"):
        return 45
    elif date <= pd.Timestamp("2023-06-30"):
        return 46
    elif date <= pd.Timestamp("2024-07-31"):
        return 48
    else:
        return 50

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df = df.dropna()
    df = df.sort_values(by='Fecha')
    return df

def build_features_targets(df: pd.DataFrame, history_window: int = 10):
    X, y = [], []

    for i in range(history_window, len(df)):
        target_date = df.iloc[i]["Fecha"]
        max_number = get_max_number_for_date(target_date)

        recent = df.iloc[i - history_window:i]

        freq_vector = np.zeros(50, dtype=int)
        for idx, (_, row) in enumerate(recent.iterrows()):
            for n in row[1:]:
                if 1 <= n <= max_number:
                    peso = idx + 1
                    freq_vector[n - 1] += peso

        y_vector = np.zeros(50, dtype=int)
        for n in df.iloc[i][1:]:
            if 1 <= n <= max_number:
                y_vector[n - 1] = 1

        X.append(freq_vector)
        y.append(y_vector)

    return np.array(X), np.array(y)

def build_single_feature_vector(df: pd.DataFrame, history_window: int = 10) -> np.ndarray:
    recent = df.iloc[-history_window:]
    last_date = df.iloc[-1]["Fecha"]
    max_number = get_max_number_for_date(last_date)

    freq_vector = np.zeros(50, dtype=int)
    for _, row in recent.iterrows():
        for n in row[1:]:
            if 1 <= n <= max_number:
                freq_vector[n - 1] += 1

    return freq_vector
