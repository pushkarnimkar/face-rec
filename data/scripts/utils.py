import os
import pandas as pd


def extract_from_two(dict1, dict2, key):
    if key in dict1:
        return dict1[key]
    elif key in dict2:
        return dict2[key]
    return None


def read_vehicle_frame(live_dir: str):
    _frame = pd.read_csv(os.path.join(live_dir, "vehicles.csv"))
    return
