import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_auc_score,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(path=r'c:\CS Study\Credit Scoring Model\data\raw\german_credit_data.csv'):
    df = pd.read_csv(path, index_col=0)
    return df

def preprocess(df):
    df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
    df['Checking account'] = df['Checking account'].fillna('Unknown')
    
    # Map target variable
    df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    X = df.drop(columns=['Risk'])
    y = df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    cols_to_scale = ['Age', 'Credit amount', 'Duration']

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, y_train):
    models = {}

    # Model 1: Decision Tree (Base)
    print("  [1/2] Decision Tree (Base)...")
    dt_base = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_base.fit(X_train, y_train)
    models['Decision Tree (Base)'] = dt_base

    # Model 2: Decision Tree with GridSearchCV
    print("  [2/2] Decision Tree + GridSearchCV (this may take a few seconds)...")

    param_grid = {
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    dt_grid_base = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=dt_grid_base,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"       Best params: {grid_search.best_params_}")
    print(f"       Best CV AUC: {grid_search.best_score_:.4f}")

    models['Decision Tree (tuned)'] = grid_search.best_estimator_

    return models

def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall':    recall_score(y_test, y_pred),
            'F1-Score':  f1_score(y_test, y_pred),
            'ROC-AUC':   roc_auc_score(y_test, y_proba)
        })

    results_df = pd.DataFrame(results).set_index('Model')

    print("\n" + "═" * 75)
    print("                        MODEL COMPARISON")
    print("═" * 75)
    print(results_df.round(4).to_string())
    print("═" * 75)

    print("\n🏆 Best model per metric:")
    for col in results_df.columns:
        best = results_df[col].idxmax()
        print(f"   {col:12s}: {best} ({results_df.loc[best, col]:.4f})")

    best_model = models['Decision Tree (tuned)']
    y_pred_best = best_model.predict(X_test)

    print("\n" + "═" * 55)
    print("     BEST MODEL: Decision Tree (tuned) — Detailed Report")
    print("═" * 55)
    print(classification_report(y_test, y_pred_best,
                                target_names=['Bad Risk (0)', 'Good Risk (1)']))

    importance = pd.Series(best_model.feature_importances_,
                           index=X_test.columns)
    importance = importance.sort_values(ascending=False)

    print("🔑 Feature Importance (top to bottom):")
    for i, (feat, val) in enumerate(importance.items(), 1):
        if val > 0:
            bar = '█' * int(val * 50)
            print(f"   {i:2d}. {feat:25s} {val:.3f}  {bar}")

    # Generate and save figures
    print("\n   Generating figures...")
    os.makedirs(r"c:\CS Study\Credit Scoring Model\figures", exist_ok=True)
    
    # 1. Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values[:10], y=importance.index[:10], palette="viridis")
    plt.title("Top 10 Feature Importances (Tuned Decision Tree)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(r"c:\CS Study\Credit Scoring Model\figures\04_feature_importance.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad Risk', 'Good Risk'], yticklabels=['Bad Risk', 'Good Risk'])
    plt.title("Confusion Matrix (Best Model)")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(r"c:\CS Study\Credit Scoring Model\figures\03_confusion_matrix_best.png", dpi=300)
    plt.close()

    # 3. ROC Curves All Models
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"c:\CS Study\Credit Scoring Model\figures\02_roc_curves_all.png", dpi=300)
    plt.close()

    return results_df

def generate_eda_figures(df):
    os.makedirs(r"c:\CS Study\Credit Scoring Model\figures", exist_ok=True)
    print("\n   Generating EDA correlations...")
    # Map target for heatmap if present as strings
    if df['Risk'].dtype == 'object':
        df_temp = df.copy()
        df_temp['Risk'] = df_temp['Risk'].map({'good': 1, 'bad': 0})
    else:
        df_temp = df

    # We use numeric columns only for correlation
    numeric_df = df_temp.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        plt.savefig(r"c:\CS Study\Credit Scoring Model\figures\01_correlation_heatmap.png", dpi=300)
        plt.close()


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  💳 Credit Scoring Prediction")
    print("=" * 50)

    print("\n📂 Loading data...")
    df = load_data()
    print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    df_temp = pd.read_csv(r'c:\CS Study\Credit Scoring Model\data\raw\german_credit_data.csv', index_col=0)
    print(f"   Good Risk rate: {(df_temp['Risk'] == 'good').mean()*100:.1f}%")

    print("\n🔧 Preprocessing...")
    generate_eda_figures(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} samples")

    print("\n🏋️ Training models...")
    models = train_models(X_train, y_train)
    print(f"   ✅ {len(models)} models trained")

    print("\n📊 Evaluating...")
    results = evaluate_models(models, X_test, y_test)

    print("\n✅ Done!")
