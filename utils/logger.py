import pandas as pd

def log_entry(path, timestamp, vehicle_type, plate_text):
    df = pd.DataFrame([[timestamp.strftime("%Y-%m-%d %H:%M:%S"), vehicle_type, plate_text]],
                      columns=["timestamp", "vehicle_type", "plate_text"])
    df.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)
