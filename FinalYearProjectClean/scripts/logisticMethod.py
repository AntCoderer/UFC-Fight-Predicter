import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

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


def get_method_class(row):
    if row.get("winKo", 0) == 1:
        return "KO/TKO"
    elif row.get("winSub", 0) == 1:
        return "Submission"
    else:
        return "Decision"

df["methodClass"] = df.apply(get_method_class, axis=1)


y_win = df["redWin"]
y_method = df["methodClass"]


drop_cols = [
    "redWin",
    "winKo",
    "winSub",
    "winDecUnanimous",
    "winDecSplit",
    "winDecMajority",
    "methodClass"
]

X = df.drop(columns=drop_cols, errors="ignore")


XTrain, XTest, yWinTrain, yWinTest, yMethodTrain, yMethodTest = train_test_split(
    X,
    y_win,
    y_method,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train size:", len(XTrain))
print("Test size :", len(XTest))


scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)


C_values = [0.01, 0.1, 1, 5, 10]

best_auc = 0
best_win_model = None
best_win_config = None

print("\nSearching best logistic model\n")

for C in C_values:

    model_l2 = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000
    )

    model_l2.fit(XTrainScaled, yWinTrain)

    probs = model_l2.predict_proba(XTestScaled)[:, 1]
    auc = roc_auc_score(yWinTest, probs)

    print(f"C={C}, L2 -> AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_win_model = model_l2
        best_win_config = (C, "l2")

    model_l1 = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000
    )

    model_l1.fit(XTrainScaled, yWinTrain)

    probs = model_l1.predict_proba(XTestScaled)[:, 1]
    auc = roc_auc_score(yWinTest, probs)

    print(f"C={C}, L1 -> AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_win_model = model_l1
        best_win_config = (C, "l1")


print("\n===== BEST WINNER MODEL =====")
print(f"C={best_win_config[0]}, type={best_win_config[1]}")
print(f"Best ROC AUC: {best_auc:.4f}")

win_probs = best_win_model.predict_proba(XTestScaled)[:, 1]
win_preds = (win_probs >= 0.5).astype(int)

print("\nWinner Accuracy (0.50 threshold):", round(accuracy_score(yWinTest, win_preds), 3))

print("\nWinner accuracy by confidence threshold:\n")

winner_thresholds = [0.5, 0.6, 0.7, 0.8]

for t in winner_thresholds:
    mask = (win_probs >= t) | (win_probs <= (1 - t))

    if mask.sum() == 0:
        print(f">{int(t*100)}% : no samples")
        continue

    filtered_probs = win_probs[mask]
    filtered_y = yWinTest.iloc[mask]
    filtered_preds = (filtered_probs >= 0.5).astype(int)

    acc = accuracy_score(filtered_y, filtered_preds)
    coverage = mask.mean()

    print(f">{int(t*100)}%  |  accuracy = {acc:.3f}  |  fights used = {mask.sum()} ({coverage:.1%})")


best_method_acc = 0
best_method_model = None
best_method_config = None

print("\nSearching best METHOD logistic model...\n")

for C in C_values:
    model_method = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000
    )

    model_method.fit(XTrainScaled, yMethodTrain)

    method_preds = model_method.predict(XTestScaled)
    acc = accuracy_score(yMethodTest, method_preds)

    print(f"C={C} -> Method ACC={acc:.4f}")

    if acc > best_method_acc:
        best_method_acc = acc
        best_method_model = model_method
        best_method_config = C


print("\n===== BEST METHOD MODEL =====")
print(f"C={best_method_config}")
print(f"Best Method Accuracy: {best_method_acc:.4f}")

all_method_preds = best_method_model.predict(XTestScaled)
all_method_probs = best_method_model.predict_proba(XTestScaled)
all_method_max_probs = all_method_probs.max(axis=1)

print("\nMethod Accuracy on ALL fights:", round(accuracy_score(yMethodTest, all_method_preds), 3))
print("\nMethod Classification Report (ALL fights):\n")
print(classification_report(yMethodTest, all_method_preds))


correct_winner_mask = (win_preds == yWinTest.values)

method_y_correct = yMethodTest.iloc[correct_winner_mask]
method_preds_correct = all_method_preds[correct_winner_mask]
method_probs_correct = all_method_max_probs[correct_winner_mask]

print("\n===== METHOD PREDICTION ONLY WHEN WINNER IS CORRECT =====")
print("Fights where predicted winner was correct:", len(method_y_correct))

if len(method_y_correct) > 0:
    method_acc_correct = accuracy_score(method_y_correct, method_preds_correct)
    print("Method Accuracy (winner correct only):", round(method_acc_correct, 3))

    print("\nClassification Report (winner correct only):\n")
    print(classification_report(method_y_correct, method_preds_correct))
else:
    print("No correct winner predictions found.")


print("\nMethod accuracy by confidence threshold (winner correct only):\n")

method_thresholds = [0.33, 0.40, 0.50, 0.60, 0.70]

if len(method_y_correct) > 0:
    for t in method_thresholds:
        mask = method_probs_correct >= t

        if mask.sum() == 0:
            print(f">{t:.2f} : no samples")
            continue

        filtered_y = method_y_correct.iloc[mask]
        filtered_preds = method_preds_correct[mask]

        acc = accuracy_score(filtered_y, filtered_preds)
        coverage = mask.mean()

        print(f">{t:.2f}  |  accuracy = {acc:.3f}  |  fights used = {mask.sum()} ({coverage:.1%})")
else:
    print("No samples available for threshold testing.")