import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from xgboost import XGBClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import argparse
import yaml

def clean_and_append(dirname, postfix):
    if dirname.endswith("/"):
        dirname = dirname[:-1]
    return dirname + postfix

def train(args):
    postfix = "" if not args.no_signal else "_no_signal"
    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    inputdir = clean_and_append(config["output"]["storedir"], "_hybrid_raw")
    inputdir = clean_and_append(inputdir, postfix)
    outputdir = clean_and_append(config["output"]["storedir"], "_score")
    os.makedirs(outputdir, exist_ok=True)


    step_dir = "03_train_cls"
    os.makedirs(os.path.join(config["output"]["plotdir"], step_dir), exist_ok=True)
    plotdir = os.path.join(config["output"]["plotdir"], step_dir)

    # Load dataset (replace with actual path)
    df = pd.read_csv(f'{inputdir}/SR/data.csv')  # You’ll need to convert the image data to CSV if not done
    df_SB = pd.read_csv(f'{inputdir}/SB/data.csv')  # You’ll need to convert the image data to CSV if not done

    # Split features and labels
    X = df.drop(columns=['classification', 'inv_mass'])
    y = df['classification'].values
    feature_names = X.columns

    # Optional: scale features
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # K-Fold setup
    knumber = args.knumber
    kf = StratifiedKFold(n_splits=knumber, shuffle=False)
    kf_SB = KFold(n_splits=knumber, shuffle=False)

    # Store ROC data
    roc_data = {'xgb': [], 'pfn': []}
    auc_scores = {'xgb': [], 'pfn': []}

    # Store original test inputs and predictions
    test_results = []
    test_results_SB = []

    for (_, test_idx_SB), (train_idx, test_idx) in zip((kf_SB.split(df_SB)),(kf.split(X_scaled, y))):
        df_SB_fold = df_SB.loc[test_idx_SB].copy()
        X_SB = scaler.transform(df_SB_fold.drop(columns=['classification', 'inv_mass']))
        print(X_SB)

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Class weight balancing
        sample_weights = compute_sample_weight('balanced', y_train)

        # === XGBoost ===
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        xgb_probs_SB = xgb.predict_proba(X_SB)[:, 1]
        df_SB_fold['xgb_prob'] = xgb_probs_SB
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
        roc_data['xgb'].append((fpr_xgb, tpr_xgb))
        auc_scores['xgb'].append(auc(fpr_xgb, tpr_xgb))

        # === AutoTabPFN ===
        pfn = AutoTabPFNClassifier(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ignore_pretraining_limits=True
        )
        pfn.fit(X_train, y_train)
        pfn_probs = pfn.predict_proba(X_test)[:, 1]
        pfn_probs_SB = pfn.predict_proba(X_SB)[:, 1]
        df_SB_fold['pfn_probs'] = pfn_probs_SB
        fpr_pfn, tpr_pfn, _ = roc_curve(y_test, pfn_probs)
        roc_data['pfn'].append((fpr_pfn, tpr_pfn))
        auc_scores['pfn'].append(auc(fpr_pfn, tpr_pfn))

        # === Store results for this fold ===
        fold_df = df.iloc[test_idx].copy()
        fold_df['xgb_prob'] =  xgb_probs
        fold_df['pfn_probs'] = pfn_probs
        test_results.append(fold_df)
        test_results_SB.append(df_SB_fold)

    # Combine all test results
    results_df = pd.concat(test_results, ignore_index=True)
    results_df_SB = pd.concat(test_results_SB, ignore_index=True)

    print(results_df)

    # --- Plot ROC Curve ---
    plt.figure(figsize=(8, 6))
    for model in ['xgb', 'pfn']:
        for i, (fpr, tpr) in enumerate(roc_data[model]):
            plt.plot(fpr, tpr, alpha=0.3, label=f"{model.upper()} Fold {i+1} AUC={auc_scores[model][i]:.2f}")
        mean_auc = np.mean(auc_scores[model])
        plt.plot([], [], ' ', label=f"Mean {model.upper()} AUC={mean_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (K-Fold)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, 'roc_curves.png'))
    plt.show()

    # --- Plot signal vs. background ---
    for col in results_df.columns:
        if col == "classification":
            continue
        plt.figure(figsize=(6, 4))
        for label in results_df['classification'].unique():
            subset = results_df[results_df['classification'] == label][col].dropna()
            plt.hist(subset, bins=50, density=True, alpha=0.5, label=str(label), histtype='step')

        plt.title(f'Density Histogram of {col} by Classification')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, f'hist_{col}.png'))
        plt.show()

    final_results = pd.concat(
        [results_df[results_df['classification']==1].drop(columns='classification'), results_df_SB],
        ignore_index=True)

    print(final_results)
    final_results.to_csv(os.path.join(outputdir, 'final_results.csv'), index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--region", type = str, default = "SR")
    parser.add_argument("--no_signal", action = "store_true", default = False)
    parser.add_argument("--knumber", type=int, default=3, help="Number of folds for K-Fold cross-validation")
    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file

    train(args)

if __name__ == "__main__":
    main()

