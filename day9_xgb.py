import os

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
    [
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(random_state=42, eval_metric="mlogloss")),
    ]
)

param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 6, 9],
    "clf__learning_rate": [0.05, 0.1, 0.2],
}

with mlflow.start_run() as run:
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("cv_acc", search.best_score_)
    mlflow.sklearn.log_model(best_model, "model", registered_model_name="PenguinXGB")
    print("Run ID:", run.info.run_id)
