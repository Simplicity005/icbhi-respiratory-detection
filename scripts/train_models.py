"""
ICBHI 2017 Respiratory Sound Database - Model Training & Benchmarking Pipeline (Phase 4).

Trains, benchmarks, and evaluates classical machine learning classifiers:
1. Random Forest (RF) - Primary baseline with cost-sensitive class balancing
2. LightGBM (LGB) - Fast gradient-boosted decision trees with balanced weighting
3. Support Vector Machine (SVM) - Non-linear RBF kernel with feature standardization
4. Logistic Regression (LR) - Standard linear baseline with L2 regularization

Target Tasks:
- 4-Class Classification: Normal (0) vs. Crackle (1) vs. Wheeze (2) vs. Both (3)
- 2-Class Classification: Normal (0) vs. Adventitious / Abnormal (1)

Validation Protocol:
- 5-Fold Patient-Aware Cross-Validation (GroupKFold grouped by patient_id on train set)
- Final independent evaluation on official ICBHI test cohort (2,756 cycles across 49 patients)
- Official ICBHI Challenge metrics:
  - Specificity (Sp): Accuracy on Normal breathing cycles
  - Sensitivity (Se): Average recall on adventitious cycles
  - ICBHI Score: (Sp + Se) / 2
  - Overall Accuracy & Macro F1-score

Exports:
- reports/model_benchmark_results.csv (Summary table of all models and metrics)
- reports/figures/fig11_model_benchmark_comparison.png (Performance comparison bar plot)
- reports/figures/fig12_confusion_matrices.png (Normalized confusion matrices)
- reports/figures/fig13_feature_importances.png (Top 20 acoustic biomarkers)
- models/ (*.joblib trained model artifacts)
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------
# Metric Calculations Following ICBHI 2017 Challenge Formulas
# ---------------------------------------------------------------------

def compute_icbhi_metrics(y_true, y_pred, is_binary=False):
    """
    Computes official ICBHI challenge metrics:
    - Specificity (Sp): Recall of Normal class (class 0)
    - Sensitivity (Se): Average recall of adventitious classes (classes 1..N)
    - Score: (Sp + Se) / 2
    """
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = cm.diagonal() / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)

    sp = float(recalls[0])  # Normal class recall

    if is_binary:
        se = float(recalls[1]) if len(recalls) > 1 else 0.0
    else:
        # Average recall across all abnormal classes (Crackle, Wheeze, Both)
        se = float(np.mean(recalls[1:])) if len(recalls) > 1 else 0.0

    score = float((sp + se) / 2.0)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    return {
        "specificity": sp,
        "sensitivity": se,
        "icbhi_score": score,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_recall": recalls
    }

# ---------------------------------------------------------------------
# Cross-Validation & Model Training
# ---------------------------------------------------------------------

def run_patient_grouped_cv(clf, X, y, patient_ids, is_binary=False, n_splits=5, scaler=None):
    """
    Perform 5-Fold Cross-Validation grouped strictly by patient_id.
    Zero patient leakage between folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = []
    cv_sp = []
    cv_se = []
    cv_acc = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=patient_ids)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if scaler is not None:
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X_tr)
            X_val = fold_scaler.transform(X_val)

        clf_fold = clone(clf)
        clf_fold.fit(X_tr, y_tr)
        preds_val = clf_fold.predict(X_val)

        metrics = compute_icbhi_metrics(y_val, preds_val, is_binary=is_binary)
        cv_scores.append(metrics["icbhi_score"])
        cv_sp.append(metrics["specificity"])
        cv_se.append(metrics["sensitivity"])
        cv_acc.append(metrics["accuracy"])

    return {
        "cv_score_mean": np.mean(cv_scores),
        "cv_score_std": np.std(cv_scores),
        "cv_sp_mean": np.mean(cv_sp),
        "cv_se_mean": np.mean(cv_se),
        "cv_acc_mean": np.mean(cv_acc)
    }

# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------

def generate_benchmark_figures(results_df, best_models, X_test, y_test, scaler, figures_dir, feature_names):
    """
    Generates Figures 11, 12, and 13 for Phase 4:
    - fig11_model_benchmark_comparison.png
    - fig12_confusion_matrices.png
    - fig13_feature_importances.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Figure 11: Benchmark Metric Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (task_name, ax) in enumerate([("4-Class Task", axes[0]), ("2-Class (Binary) Task", axes[1])]):
        sub_df = results_df[results_df["task"] == task_name].copy()
        plot_data = sub_df.melt(
            id_vars=["model"],
            value_vars=["icbhi_score", "sensitivity", "specificity", "accuracy"],
            var_name="Metric",
            value_name="Score"
        )
        plot_data["Score"] = plot_data["Score"] * 100
        metric_labels = {
            "icbhi_score": "ICBHI Score",
            "sensitivity": "Sensitivity (Se)",
            "specificity": "Specificity (Sp)",
            "accuracy": "Accuracy"
        }
        plot_data["Metric"] = plot_data["Metric"].map(metric_labels)

        palette = {"ICBHI Score": "#1971c2", "Sensitivity (Se)": "#d9480f", "Specificity (Sp)": "#2b8a3e", "Accuracy": "#7048e8"}
        sns.barplot(data=plot_data, x="model", y="Score", hue="Metric", palette=palette, ax=ax, edgecolor="black", linewidth=0.7)
        ax.set_title(f"{task_name} - Model Performance on Official Test Set", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model", fontweight="bold")
        ax.set_ylabel("Score (%)", fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle("ICBHI 2017 Benchmark Evaluation across Machine Learning Baselines", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig11_path = os.path.join(figures_dir, "fig11_model_benchmark_comparison.png")
    fig.savefig(fig11_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 11: {fig11_path}")

    # 2. Figure 12: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 4-Class Best Model (Random Forest or LightGBM)
    rf_4c = best_models["rf_4class"]
    preds_4c = rf_4c.predict(X_test)
    cm_4c = confusion_matrix(y_test["class_id"], preds_4c, normalize='true') * 100
    class_labels_4c = ["Normal", "Crackle", "Wheeze", "Both"]

    sns.heatmap(cm_4c, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=class_labels_4c, yticklabels=class_labels_4c, ax=axes[0],
                linewidths=0.5, linecolor="gray", annot_kws={"size": 11, "fontweight": "bold"})
    axes[0].set_title("Random Forest: 4-Class Normalized Confusion Matrix (%)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Class", fontweight="bold")
    axes[0].set_ylabel("True Clinical Label", fontweight="bold")

    # Binary Best Model (LightGBM or Random Forest)
    lgb_bin = best_models["lgb_binary"]
    preds_bin = lgb_bin.predict(X_test)
    cm_bin = confusion_matrix(y_test["is_adventitious"], preds_bin, normalize='true') * 100
    class_labels_bin = ["Normal", "Adventitious"]

    sns.heatmap(cm_bin, annot=True, fmt=".1f", cmap="Oranges", cbar=True,
                xticklabels=class_labels_bin, yticklabels=class_labels_bin, ax=axes[1],
                linewidths=0.5, linecolor="gray", annot_kws={"size": 11, "fontweight": "bold"})
    axes[1].set_title("LightGBM: Binary Normalized Confusion Matrix (%)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Predicted Class", fontweight="bold")
    axes[1].set_ylabel("True Clinical Label", fontweight="bold")

    fig.suptitle("Normalized Confusion Matrices on Official ICBHI Test Cohort (N = 2,756 cycles)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig12_path = os.path.join(figures_dir, "fig12_confusion_matrices.png")
    fig.savefig(fig12_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 12: {fig12_path}")

    # 3. Figure 13: Feature Importances (Top 20)
    rf_4c = best_models["rf_4class"]
    importances = rf_4c.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    top20 = feat_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top20, x="Importance", y="Feature", hue="Feature", palette="viridis", ax=ax, edgecolor="black", linewidth=0.6, legend=False)
    ax.set_title("Top 20 Most Predictive Acoustic Biomarkers (Random Forest 4-Class)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontweight="bold")
    ax.set_ylabel("Acoustic Biomarker", fontweight="bold")
    plt.tight_layout()
    fig13_path = os.path.join(figures_dir, "fig13_feature_importances.png")
    fig.savefig(fig13_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 13: {fig13_path}")

# ---------------------------------------------------------------------
# Main Model Training & Benchmarking Pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ICBHI Model Training and Benchmarking Pipeline")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of GroupKFold splits (default: 5)")
    args = parser.parse_args()

    t_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = os.path.join(project_root, "data", "processed")
    reports_dir = os.path.join(project_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    models_dir = os.path.join(project_root, "models")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 75)
    print("      ICBHI 2017 MACHINE LEARNING TRAINING & BENCHMARKING (PHASE 4)")
    print("=" * 75)
    print(f"Project Root    : {project_root}")
    print(f"Data Directory  : {data_dir}")
    print(f"Models Saved To : {models_dir}")
    print(f"Reports Output  : {reports_dir}")
    print(f"CV Protocol     : {args.cv_folds}-Fold Patient-Grouped (Zero Patient Leakage)")
    print("=" * 75)

    # 1. Load Datasets
    print("\n[1/5] Loading pre-extracted feature benchmark matrices...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    meta_train = pd.read_csv(os.path.join(data_dir, "metadata_train.csv"))

    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    meta_test = pd.read_csv(os.path.join(data_dir, "metadata_test.csv"))

    feature_names = list(X_train.columns)
    patient_ids_train = meta_train["patient_id"].values

    print(f"Train Cohort: {X_train.shape[0]:,} cycles x {X_train.shape[1]} features (79 patients)")
    print(f"Test Cohort : {X_test.shape[0]:,} cycles x {X_test.shape[1]} features (49 patients)")

    # Standardize features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # 2. Define Model Configurations
    models = {
        "Random Forest": {
            "model_4c": RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1, class_weight='balanced'),
            "model_bin": RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1, class_weight='balanced'),
            "use_scaled": False
        },
        "LightGBM": {
            "model_4c": LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
            "model_bin": LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
            "use_scaled": False
        },
        "Support Vector Machine": {
            "model_4c": SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42),
            "model_bin": SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=42),
            "use_scaled": True
        },
        "Logistic Regression": {
            "model_4c": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "model_bin": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "use_scaled": True
        }
    }

    benchmark_rows = []
    best_fitted_models = {}

    # 3. Train & Evaluate Models across Tasks
    print("\n[2/5] Training and benchmarking classifiers on 4-Class Task...")
    for model_name, cfg in models.items():
        t0 = time.time()
        clf_4c = cfg["model_4c"]
        use_scaled = cfg["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train.values
        X_te = X_test_scaled if use_scaled else X_test.values
        y_tr_4c = y_train["class_id"].values
        y_te_4c = y_test["class_id"].values

        # Patient-grouped CV
        print(f"  Training {model_name} (4-Class)...")
        cv_res = run_patient_grouped_cv(
            clf_4c,
            pd.DataFrame(X_tr) if isinstance(X_tr, np.ndarray) else X_train,
            pd.Series(y_tr_4c),
            patient_ids_train,
            is_binary=False,
            n_splits=args.cv_folds
        )

        # Fit on full training set
        clf_4c.fit(X_tr, y_tr_4c)
        preds_test = clf_4c.predict(X_te)
        test_metrics = compute_icbhi_metrics(y_te_4c, preds_test, is_binary=False)
        elapsed = time.time() - t0

        print(f"    -> Test ICBHI Score: {test_metrics['icbhi_score']*100:5.2f}% | "
              f"Se: {test_metrics['sensitivity']*100:5.2f}% | "
              f"Sp: {test_metrics['specificity']*100:5.2f}% | "
              f"Acc: {test_metrics['accuracy']*100:5.2f}% | Elapsed: {elapsed:4.1f}s")

        benchmark_rows.append({
            "task": "4-Class Task",
            "model": model_name,
            "cv_icbhi_score": cv_res["cv_score_mean"],
            "cv_icbhi_std": cv_res["cv_score_std"],
            "icbhi_score": test_metrics["icbhi_score"],
            "sensitivity": test_metrics["sensitivity"],
            "specificity": test_metrics["specificity"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "train_time_sec": elapsed
        })

        if model_name == "Random Forest":
            best_fitted_models["rf_4class"] = clf_4c
            joblib.dump(clf_4c, os.path.join(models_dir, "random_forest_4class.joblib"))
        elif model_name == "LightGBM":
            best_fitted_models["lgb_4class"] = clf_4c
            joblib.dump(clf_4c, os.path.join(models_dir, "lightgbm_4class.joblib"))

    print("\n[3/5] Training and benchmarking classifiers on 2-Class (Binary) Task...")
    for model_name, cfg in models.items():
        t0 = time.time()
        clf_bin = cfg["model_bin"]
        use_scaled = cfg["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train.values
        X_te = X_test_scaled if use_scaled else X_test.values
        y_tr_bin = y_train["is_adventitious"].values
        y_te_bin = y_test["is_adventitious"].values

        print(f"  Training {model_name} (Binary)...")
        cv_res = run_patient_grouped_cv(
            clf_bin,
            pd.DataFrame(X_tr) if isinstance(X_tr, np.ndarray) else X_train,
            pd.Series(y_tr_bin),
            patient_ids_train,
            is_binary=True,
            n_splits=args.cv_folds
        )

        clf_bin.fit(X_tr, y_tr_bin)
        preds_test = clf_bin.predict(X_te)
        test_metrics = compute_icbhi_metrics(y_te_bin, preds_test, is_binary=True)
        elapsed = time.time() - t0

        print(f"    -> Test ICBHI Score: {test_metrics['icbhi_score']*100:5.2f}% | "
              f"Se: {test_metrics['sensitivity']*100:5.2f}% | "
              f"Sp: {test_metrics['specificity']*100:5.2f}% | "
              f"Acc: {test_metrics['accuracy']*100:5.2f}% | Elapsed: {elapsed:4.1f}s")

        benchmark_rows.append({
            "task": "2-Class (Binary) Task",
            "model": model_name,
            "cv_icbhi_score": cv_res["cv_score_mean"],
            "cv_icbhi_std": cv_res["cv_score_std"],
            "icbhi_score": test_metrics["icbhi_score"],
            "sensitivity": test_metrics["sensitivity"],
            "specificity": test_metrics["specificity"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "train_time_sec": elapsed
        })

        if model_name == "LightGBM":
            best_fitted_models["lgb_binary"] = clf_bin
            joblib.dump(clf_bin, os.path.join(models_dir, "lightgbm_binary.joblib"))
        elif model_name == "Random Forest":
            best_fitted_models["rf_binary"] = clf_bin
            joblib.dump(clf_bin, os.path.join(models_dir, "random_forest_binary.joblib"))
        elif model_name == "Support Vector Machine":
            joblib.dump(clf_bin, os.path.join(models_dir, "svm_binary.joblib"))

    # 4. Save Benchmark Results Table
    print("\n[4/5] Exporting benchmark results summary...")
    results_df = pd.DataFrame(benchmark_rows)
    results_csv_path = os.path.join(reports_dir, "model_benchmark_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved Results Table: {results_csv_path}")

    # Display clean formatted table
    display_df = results_df.copy()
    for col in ["cv_icbhi_score", "icbhi_score", "sensitivity", "specificity", "accuracy", "macro_f1"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    display_df["train_time_sec"] = display_df["train_time_sec"].round(1).astype(str) + "s"
    print("\n" + display_df[["task", "model", "icbhi_score", "sensitivity", "specificity", "accuracy", "macro_f1", "train_time_sec"]].to_string(index=False))

    # 5. Generate Phase 4 Verification Figures
    print("\n[5/5] Generating publication-ready figures (Figures 11, 12, 13)...")
    generate_benchmark_figures(
        results_df,
        best_fitted_models,
        X_test.values,
        y_test,
        scaler,
        figures_dir,
        feature_names
    )

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"  PHASE 4 MODEL BENCHMARKING COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 75)

if __name__ == "__main__":
    main()

