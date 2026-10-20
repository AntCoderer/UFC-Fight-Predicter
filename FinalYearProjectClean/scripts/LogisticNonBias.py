import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from pathlib import Path


# Load data

basePath = Path(__file__).resolve().parents[1]
dataPath = basePath / "data"
csvPath = dataPath / "training_dataset_mk3.csv"

print("Loading:", csvPath)

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





X_orig = df.drop(columns=["redWin"])
y_orig = df["redWin"]

X_swap = X_orig.copy()

for col in X_orig.columns:
    if col.startswith("red"):
        X_swap["blue" + col[3:]] = X_orig[col]
    elif col.startswith("blue"):
        X_swap["red" + col[4:]] = X_orig[col]

X_swap = X_swap[X_orig.columns]
y_swap = 1 - y_orig

# Combine both
X = pd.concat([X_orig, X_swap], axis=0).reset_index(drop=True)
y = pd.concat([y_orig, y_swap], axis=0).reset_index(drop=True)





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








import joblib
import os

modelDir = basePath / "models"
os.makedirs(modelDir, exist_ok=True)


modelPath = modelDir / "logistic_model_mk3.pkl"
joblib.dump(best_model, modelPath)
print("\nModel saved to:", modelPath)


featurePath = modelDir / "features.pkl"
joblib.dump(list(X.columns), featurePath)


scalerPath = modelDir / "scaler.pkl"
joblib.dump(scaler, scalerPath)


coef = best_model.coef_[0]

feat_df = pd.DataFrame({
    "feature": X.columns,
    "importance": np.abs(coef) 
}).sort_values(by="importance", ascending=False)

importancePath = modelDir / "feature_importance.pkl"
joblib.dump(feat_df, importancePath)

print("Feature importance saved to:", importancePath)





def predict_fight(fighter_row_dict):

    model = joblib.load(modelPath)
    feature_names = joblib.load(featurePath)
    scaler = joblib.load(scalerPath)
    importance = joblib.load(importancePath)


    df_input = pd.DataFrame([fighter_row_dict])

    for col in feature_names:
        if col not in df_input:
            df_input[col] = 0

    df_input = df_input[feature_names]
    df_input_scaled = scaler.transform(df_input)

    prob1 = model.predict_proba(df_input_scaled)[0][1]


    swapped = {}
    for k, v in fighter_row_dict.items():
        if k.startswith("red"):
            swapped["blue" + k[3:]] = v
        elif k.startswith("blue"):
            swapped["red" + k[4:]] = v
        else:
            swapped[k] = v

    df_swap = pd.DataFrame([swapped])

    for col in feature_names:
        if col not in df_swap:
            df_swap[col] = 0

    df_swap = df_swap[feature_names]
    df_swap_scaled = scaler.transform(df_swap)

    prob2 = model.predict_proba(df_swap_scaled)[0][1]


    prob = (prob1 + (1 - prob2)) / 2
    pred = int(prob >= 0.5)

    return {
        "predictedWinner": "Red" if pred == 1 else "Blue",
        "redWinProbability": float(prob),
        "confidence": float(max(prob, 1 - prob)),
        "topFeatures": importance.head(5).to_dict(orient="records")
    }