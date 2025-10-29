import os

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# DAGsHub credentials
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/liqinsg/ai-sprint.mlflow"
os.environ["DAGSHUB_USERNAME"] = "liqinsg"
os.environ["DAGSHUB_PASSWORD"] = "ff0b57740387f5904dbd095c0ff04d182237d0f7"

# === same Day 5 code ===
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url).dropna(
    subset=[
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
)
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))]
)
param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 4],
}

with mlflow.start_run() as run:
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("cv_acc", search.best_score_)
    mlflow.sklearn.log_model(best_model, "model", registered_model_name="PenguinRF")
    print("Run ID:", run.info.run_id)
