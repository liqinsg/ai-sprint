# day8_basic_auth.py
import os

import mlflow

print("URI:", os.environ.get("MLFLOW_TRACKING_URI"))
print("USER:", os.environ.get("DAGSHUB_USERNAME"))
print("TOKEN:", os.environ.get("DAGSHUB_PASSWORD")[:4] + "****")

# plain Basic-Auth – no browser pop-up
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/liqinsg/ai-sprint.mlflow"
os.environ["DAGSHUB_USERNAME"] = "liqinsg"
os.environ["DAGSHUB_PASSWORD"] = "8f91a3bd49d6b3c64fc61baf6b9a0aac5dbf1a3f"

with mlflow.start_run() as run:
    mlflow.log_metric("dummy", 1.0)
    print("Run ID:", run.info.run_id)
