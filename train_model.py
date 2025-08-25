"""
Train a severity classifier on processed data:
- Reads data/processed/accidents_clean.csv
- Encodes categorical features
- Trains RandomForest & evaluates
- Saves model to models/severity_model.joblib
- Saves metrics to models/metrics.json
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

DATA_PATH = Path("data/processed/accidents_clean.csv")
MODEL_PATH = Path("models/severity_model.joblib")
METRICS_PATH = Path("models/metrics.json")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Define features/target
    target = "severity"
    # Simple, editable feature list
    features = ["state","city","vehicle_type","weather","road_type","year","month","weekday","time_of_day"]
    features = [f for f in features if f in df.columns]

    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]

    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")

    model = Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "report": report, "features": features}, f, indent=2)

    print(f"Saved model → {MODEL_PATH}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Metrics JSON → {METRICS_PATH}")

if __name__ == "__main__":
    main()
