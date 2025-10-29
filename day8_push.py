import os

import dagshub
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/liqinsg/ai-sprint.mlflow"
os.environ["DAGSHUB_USERNAME"] = "liqinsg"
os.environ["DAGSHUB_PASSWORD"] = "ff0b57740387f5904dbd095c0ff04d182237d0f7"

dagshub.init(repo_owner="liqinsg", repo_name="ai-sprint")  # no token arg

with mlflow.start_run():
    mlflow.log_metric("dummy", 1.0)
print("Pushed first run – endpoint now exists!")
