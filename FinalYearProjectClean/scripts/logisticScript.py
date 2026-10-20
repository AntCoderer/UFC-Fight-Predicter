import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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



C_values = [0.01, 0.1, 1, 5, 10]

best_auc = 0
best_model = None
best_config = None

print("\nSearching best logistic model...\n")

for C in C_values:


    model_l2 = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000
    )

    model_l2.fit(XTrain, yTrain)

    probs = model_l2.predict_proba(XTest)[:, 1]
    auc = roc_auc_score(yTest, probs)

    print(f"C={C}, L2 -> AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model_l2
        best_config = (C, "l2")


    model_l1 = LogisticRegression(
        C=C,
        solver="liblinear",
        penalty="l1",
        max_iter=5000
    )

    model_l1.fit(XTrain, yTrain)

    probs = model_l1.predict_proba(XTest)[:, 1]
    auc = roc_auc_score(yTest, probs)

    print(f"C={C}, L1 -> AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model_l1
        best_config = (C, "l1")



print("\n===== BEST MODEL =====")
print(f"C={best_config[0]}, type={best_config[1]}")
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