import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier

from pathlib import Path


basePath = Path(__file__).resolve().parents[1]
dataPath = basePath / "data"
csvPath = dataPath / "training_dataset_mk3.csv"



df = pd.read_csv(csvPath)

print("Rows:", len(df))
print("Columns:", len(df.columns))



stanceMap = {
    "Orthodox": 0,
    "Southpaw": 1,
    "Switch": 2,
    "Open Stance": 3
}

if "redStance" in df.columns:
    df["redStance"] = df["redStance"].map(stanceMap).fillna(0)

if "blueStance" in df.columns:
    df["blueStance"] = df["blueStance"].map(stanceMap).fillna(0)


df = df.drop(columns=["redName", "blueName"], errors="ignore")


df = df.apply(pd.to_numeric, errors="coerce").fillna(0)



y = df["redWin"]
leakage_cols = [
    "redWin",
    "winKo",
    "winSub",
    "winDecUnanimous",
    "winDecSplit",
    "winDecMajority"
]

X = df.drop(columns=leakage_cols, errors="ignore")



XTrain, XTest, yTrain, yTest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train size:", len(XTrain))
print("Test size :", len(XTest))



scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)



n_estimators_list = [100, 200]
max_depth_list = [3, 5, 7]
learning_rates = [0.05, 0.1]

best_auc = 0
best_model = None
best_config = None

print("\nSearching best XGBoost model...\n")

for n in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rates:

            model = XGBClassifier(
                n_estimators=n,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )

            model.fit(XTrain, yTrain)

            probs = model.predict_proba(XTest)[:, 1]
            auc = roc_auc_score(yTest, probs)

            print(f"n={n}, depth={depth}, lr={lr} -> AUC={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_config = (n, depth, lr)



print("\n===== BEST MODEL =====")
print(f"n_estimators={best_config[0]}")
print(f"max_depth={best_config[1]}")
print(f"learning_rate={best_config[2]}")
print(f"Best ROC AUC: {best_auc:.4f}")

probs = best_model.predict_proba(XTest)[:, 1]
preds = (probs >= 0.5).astype(int)

print("\nAccuracy (0.50 threshold):", round(accuracy_score(yTest, preds), 3))



thresholds = [0.5, 0.6, 0.7, 0.8]

print("\nAccuracy by confidence threshold:\n")

for t in thresholds:

    mask = (probs >= t) | (probs <= (1 - t))

    if mask.sum() == 0:
        print(f">{int(t*100)}% : no samples")
        continue

    filtered_probs = probs[mask]
    filtered_y = yTest.iloc[mask]

    filtered_preds = (filtered_probs >= 0.5).astype(int)

    acc = accuracy_score(filtered_y, filtered_preds)
    coverage = mask.mean()

    print(
        f">{int(t*100)}%  |  accuracy = {acc:.3f}  |  fights used = {mask.sum()} ({coverage:.1%})"
    )



importances = best_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 15 Feature Importances:\n")
print(importance_df.head(15))