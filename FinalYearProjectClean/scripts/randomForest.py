import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from pathlib import Path






basePath = Path(__file__).resolve().parents[1]
dataPath = basePath / "data"
csvPath = dataPath / "training_dataset_mk3.csv"



df = pd.read_csv(csvPath)

print("Rows:", len(df))
print("Columns:", len(df.columns))




# make stance map as it cant read text
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


# drops names for same reason as above
df = df.drop(columns=["redName", "blueName"], errors="ignore")

# this part removes method as this counts as data leakage
leakage_cols = [
    "winMethod",
    "methodKO",
    "methodSub",
    "methodDecision",
    "finishRound",
    "finishTime",
    "winKo",
    "winSub",
    "winDecUnanimous",
    "winDecSplit",
    "winDecMajority"
]
df = df.drop(columns=leakage_cols, errors="ignore")


# numeric safety
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)


# sets y to the target or result and x to all the training features
y = df["redWin"]
X = df.drop(columns=["redWin"])


# Split
# this section splits the data given into training and testing
# 80/20 split with shuffle to prevent ordering bias
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train size:", len(XTrain))
print("Test size :", len(XTest))


# random forest search 

# number of trees
n_estimators_list = [100, 200, 300]

# max tree depth
max_depth_list = [None, 10, 15]

# minimum samples to split a node
min_samples_split_list = [2, 5, 10]

# minimum samples to split a node
min_samples_leaf_list = [1, 2, 4]



best_auc = 0
best_model = None
best_config = None


print("\nSearching best Random Forest...\n")


# the loop iterates through different combinations of random forest hyperparameters
# training a model for each configuration and evaluating its performance using ROC AUC
# the best performing model is selected based on the highest AUC score

for n in n_estimators_list:
    for depth in max_depth_list:
        for split in min_samples_split_list:
            for leaf in min_samples_leaf_list:

                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced" 
                )

                model.fit(XTrain, yTrain)

                probs = model.predict_proba(XTest)[:, 1]
                auc = roc_auc_score(yTest, probs)

                print(
                    f"n={n}, depth={depth}, split={split}, leaf={leaf} -> AUC={auc:.4f}"
                )

                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_config = (n, depth, split, leaf)




print("\n===== BEST MODEL =====")
print(f"n_estimators={best_config[0]}")
print(f"max_depth={best_config[1]}")
print(f"min_samples_split={best_config[2]}")
print(f"min_samples_leaf={best_config[3]}")
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



# FEATURE IMPORTANCE
print("\nTop 15 Feature Importances:\n")

importances = best_model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feat_df = feat_df.sort_values(by="importance", ascending=False)

print(feat_df.head(15))










