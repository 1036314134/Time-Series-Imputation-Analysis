from pathlib import Path

import pandas as pd


def load_data_one_series(csv_relative_path):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / csv_relative_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError("CSV file must contain at least two columns.")

    return df.iloc[:, [0, -1]].copy()