"""
Supervised ML Assignment 4 — ShopSmart Purchase Intent Prediction
Decision Tree Classifier with Pruning | Target metric: F1 score (benchmark = 0.55)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("1785251361201_shop_smart_ecommerce.csv")
print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------

# 2.1 Target distribution -> shows the imbalance
print("\nRevenue distribution:\n", df['Revenue'].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(x='Revenue', data=df)
plt.title("Revenue (Target) Distribution")
plt.savefig("eda_target_distribution.png", bbox_inches='tight')
plt.close()

# 2.2 Numerical feature distributions
num_cols = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

df[num_cols].hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.savefig("eda_numeric_histograms.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.title("Boxplots — Outlier Check")
plt.savefig("eda_boxplots.png", bbox_inches='tight')
plt.close()

# 2.3 Categorical vs Revenue
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x='Month', hue='Revenue', data=df, ax=axes[0],
              order=['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'])
axes[0].set_title("Month vs Revenue")
sns.countplot(x='VisitorType', hue='Revenue', data=df, ax=axes[1])
axes[1].set_title("VisitorType vs Revenue")
sns.countplot(x='Weekend', hue='Revenue', data=df, ax=axes[2])
axes[2].set_title("Weekend vs Revenue")
plt.tight_layout()
plt.savefig("eda_categorical_vs_revenue.png", bbox_inches='tight')
plt.close()

# 2.4 Correlation heatmap (numeric features only)
plt.figure(figsize=(10, 8))
corr = df[num_cols + ['OperatingSystems', 'Browser', 'Region', 'TrafficType']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("eda_correlation_heatmap.png", bbox_inches='tight')
plt.close()

print("\nEDA plots saved as PNGs in working directory.")

# ---------------------------------------------------------------
# 3. FEATURE PREPROCESSING
# ---------------------------------------------------------------

data = df.copy()

# 3.1 Boolean -> int
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

# 3.2 Treat OperatingSystems, Browser, Region, TrafficType as CATEGORICAL
#     (they are encoded integers, not ordinal/continuous quantities)
categorical_cols = ['Month', 'VisitorType', 'OperatingSystems',
                     'Browser', 'Region', 'TrafficType']

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 3.3 Split features / target
X = data.drop(columns=['Revenue'])
y = data['Revenue']

# 3.4 Train-test split — stratify to preserve class imbalance ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train Revenue ratio:\n", y_train.value_counts(normalize=True))
print("Test Revenue ratio:\n", y_test.value_counts(normalize=True))

# ---------------------------------------------------------------
# 4. BASELINE DECISION TREE (no pruning) — check overfitting
# ---------------------------------------------------------------
base_tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
base_tree.fit(X_train, y_train)

train_f1 = f1_score(y_train, base_tree.predict(X_train))
test_f1 = f1_score(y_test, base_tree.predict(X_test))
print(f"\n[Baseline Tree] Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")
print(f"Tree depth: {base_tree.get_depth()}, Leaves: {base_tree.get_n_leaves()}")
# Large gap between train and test F1 => overfitting => pruning needed

# ---------------------------------------------------------------
# 5. PRUNING
# ---------------------------------------------------------------

# 5.1 Pre-pruning via GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
}

grid = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid, scoring='f1', cv=5, n_jobs=-1
)
grid.fit(X_train, y_train)

print("\n[Pre-pruning] Best params:", grid.best_params_)
best_pre_pruned = grid.best_estimator_
test_f1_pre = f1_score(y_test, best_pre_pruned.predict(X_test))
print(f"[Pre-pruned Tree] Test F1: {test_f1_pre:.4f}")

# 5.2 Post-pruning via Cost Complexity Pruning (ccp_alpha)
path = base_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Sample alphas to keep runtime reasonable (too many unique values otherwise)
ccp_alphas_sampled = ccp_alphas[::max(1, len(ccp_alphas)//50)]

f1_scores_by_alpha = []
for alpha in ccp_alphas_sampled:
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=42, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    f1_scores_by_alpha.append(f1_score(y_test, clf.predict(X_test)))

best_alpha_idx = int(np.argmax(f1_scores_by_alpha))
best_alpha = ccp_alphas_sampled[best_alpha_idx]
print(f"\n[Post-pruning] Best ccp_alpha: {best_alpha:.6f}")

best_post_pruned = DecisionTreeClassifier(
    class_weight='balanced', random_state=42, ccp_alpha=best_alpha
)
best_post_pruned.fit(X_train, y_train)
test_f1_post = f1_score(y_test, best_post_pruned.predict(X_test))
print(f"[Post-pruned Tree] Test F1: {test_f1_post:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas_sampled, f1_scores_by_alpha, marker='o')
plt.xlabel("ccp_alpha")
plt.ylabel("Test F1 score")
plt.title("F1 score vs ccp_alpha (Post-pruning)")
plt.savefig("ccp_alpha_vs_f1.png", bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------
# 6. FINAL MODEL SELECTION & EVALUATION
# ---------------------------------------------------------------
candidates = {
    "Baseline (no pruning)": (base_tree, test_f1),
    "Pre-pruned (GridSearch)": (best_pre_pruned, test_f1_pre),
    "Post-pruned (ccp_alpha)": (best_post_pruned, test_f1_post),
}

best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_f1 = candidates[best_name]

print("\n" + "=" * 50)
print("MODEL COMPARISON")
for name, (_, f1) in candidates.items():
    print(f"  {name:30s} Test F1 = {f1:.4f}")
print("=" * 50)
print(f"BEST MODEL: {best_name} | Test F1 = {best_f1:.4f}")
print(f"Benchmark F1 = 0.55 -> {'PASSED' if best_f1 >= 0.55 else 'NOT PASSED'}")

print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_model.predict(X_test)))

cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 'Purchase'])
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix — {best_name}")
plt.savefig("confusion_matrix_best_model.png", bbox_inches='tight')
plt.close()

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png", bbox_inches='tight')
plt.close()

print("\nDone. All plots saved in working directory.")
