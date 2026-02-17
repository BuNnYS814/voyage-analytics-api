import csv
from datetime import datetime
import os
from typing import Dict, Any

LOGS_DIR = "logs"
PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.csv")

os.makedirs(LOGS_DIR, exist_ok=True)


def log_prediction(input_data, prediction) -> None:
    """Append a single prediction record to the log CSV."""
    with open(PREDICTIONS_LOG, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.utcnow().isoformat(), input_data, prediction])


def get_metrics() -> Dict[str, Any]:
    """Compute simple prediction metrics from the CSV log."""
    if not os.path.exists(PREDICTIONS_LOG):
        return {
            "total_predictions": 0,
            "avg_prediction": None,
            "last_prediction_time": None,
        }

    total = 0
    sum_pred = 0.0
    last_time = None

    with open(PREDICTIONS_LOG, newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3:
                continue
            ts, _input, pred = row[0], row[1], row[2]
            try:
                total += 1
                sum_pred += float(pred)
                last_time = ts
            except ValueError:
                continue

    avg_pred = sum_pred / total if total > 0 else None

    return {
        "total_predictions": total,
        "avg_prediction": avg_pred,
        "last_prediction_time": last_time,
    }
