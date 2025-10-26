import pandas as pd, joblib, mlflow, click
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


@click.command()
@click.option("--csv", required=True, help="Path to new training CSV")
@click.option("--model-out", default="models/penguin_model_v2.pkl", help="Where to save new pipeline")
def run_retrain(csv: str, model_out: str):
    mlflow.set_experiment("ai-sprint-retrain")
    df = pd.read_csv(csv)

    # basic clean
    df = df.dropna(subset=["species","bill_length_mm","bill_depth_mm",
                           "flipper_length_mm","body_mass_g"])
    X = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # log & save
    with mlflow.start_run():
        mlflow.log_param("csv", csv)
        mlflow.log_metric("cv_acc_mean", cv_scores.mean())
        mlflow.log_metric("test_acc", test_acc)
        mlflow.sklearn.log_model(pipe, "model", input_example=X_train.iloc[[0]])
    joblib.dump(pipe, model_out)
    print(f"Retrained model saved → {model_out}  (test acc {test_acc:.3f})")


if __name__ == "__main__":
    run_retrain()

"""
curl -o penguins_big.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
python retrain.py --csv penguins_big.csv --model-out penguin_model_v2.pkl
"""
