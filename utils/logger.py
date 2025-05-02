import os
import pandas as pd
from datetime import datetime

def init_log(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "logs.csv")
    if not os.path.isfile(log_path):
        df_log = pd.DataFrame(columns=["timestamp", "vehicle_type", "plate_text"])
        df_log.to_csv(log_path, index=False)
    return log_path

def append_log(log_path, vehicle_type, plate_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_log = pd.DataFrame([[timestamp, vehicle_type, plate_text]],
                          columns=["timestamp", "vehicle_type", "plate_text"])
    df_log.to_csv(log_path, mode='a', header=False, index=False)
