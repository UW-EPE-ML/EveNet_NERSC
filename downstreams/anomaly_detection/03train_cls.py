import os

import random

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
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
    bdt_hyperparameters = {
        "n_estimators": 300,  # number of boosting stages
        "max_depth": 3,  # max depth of individual regression estimators; related to complexity
        "learning_rate": 0.1,  # stop training BDT is validation loss doesn't improve after this many rounds
        "subsample": 0.7 , # fraction of samples to be used for fitting the individual base learners
        "early_stopping_rounds": 10
    }

    drop_features = ['inv_mass', 'classification']
    if args.drop is not None:
        drop_features += args.drop

    postfix = "" if not args.no_signal else "_no_signal"
    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    inputdir = clean_and_append(config["output"]["storedir"], "_hybrid_raw")
    inputdir = clean_and_append(inputdir, postfix)
    outputdir = clean_and_append(config["output"]["storedir"], postfix)
    outputdir = clean_and_append(outputdir, "_score")
    os.makedirs(outputdir, exist_ok=True)


    step_dir = "03_train_cls"
    plotdir = os.path.join(clean_and_append(config["output"]["plotdir"], postfix), step_dir)
    os.makedirs(plotdir, exist_ok=True)

    # Load dataset (replace with actual path)
    df = pd.read_csv(f'{inputdir}/SR/data.csv')  # You’ll need to convert the image data to CSV if not done
    df_SB = pd.read_csv(f'{inputdir}/SB/data.csv')  # You’ll need to convert the image data to CSV if not done



    test_no_signal = args.test_no_signal
    if test_no_signal:
        if not args.no_signal:
            input_no_signal = clean_and_append(clean_and_append(config["output"]["storedir"], "_hybrid_raw"), "_no_signal")
            df_SR_no_signal = pd.read_csv(f'{input_no_signal}/SR/data.csv')
            df_SR_no_signal = df_SR_no_signal[df_SR_no_signal["classification"]== 1]  # Only keep signal events
            df_SB_no_signal = pd.read_csv(f'{input_no_signal}/SB/data.csv')
            df_no_signal = pd.concat([df_SR_no_signal, df_SB_no_signal], ignore_index=True)
            print(f"✅ Loaded no signal data with {df_no_signal.shape[0]} events (SB: {df_SB_no_signal.shape[0]}, SR: {df_SR_no_signal.shape[0]})")
        else:
            input_no_signal = clean_and_append(config["output"]["storedir"], "_hybrid_raw")
            df_SR_no_signal = pd.read_csv(f'{input_no_signal}/SR/data.csv')
            df_SR_no_signal = df_SR_no_signal[df_SR_no_signal["classification"]== 1]  # Only keep signal events
            df_SB_no_signal = pd.read_csv(f'{input_no_signal}/SB/data.csv')
            df_no_signal = pd.concat([df_SR_no_signal, df_SB_no_signal], ignore_index=True)
            print(f"✅ Loaded no signal data with {df_no_signal.shape[0]} events (SB: {df_SB_no_signal.shape[0]}, SR: {df_SR_no_signal.shape[0]})")

    print(df_SB)

    if args.only_pc:
        for feature_name in df.columns:
            if "pc" not in feature_name and feature_name not in drop_features:
                drop_features.append(feature_name)

    # Split features and labels
    X = df.drop(columns= drop_features)
    y = df['classification'].values
    feature_names = X.columns
    print(f"Target features: {X.columns}")

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
    test_results_no_signal = []

    kfold_index_no_signal = kf_SB.split(df_no_signal)
    kfold_index_no_signal = list(kfold_index_no_signal)

    for ifold, ((_, test_idx_SB), (train_idx, test_idx)) in enumerate(zip((kf_SB.split(df_SB)),(kf.split(X_scaled, y)))):
        df_SB_fold = df_SB.loc[test_idx_SB].copy()
        X_SB = scaler.transform(df_SB_fold.drop(columns= drop_features))

        _, test_idx_no_signal = kfold_index_no_signal[ifold]
        df_no_signal_fold = df_no_signal.loc[test_idx_no_signal].copy()
        X_no_signal = scaler.transform(df_no_signal_fold.drop(columns= drop_features))
        score_fold_no_signal = np.empty((X_no_signal.shape[0], args.n_ensemble))

        X_train_full, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        score_fold = np.empty((X_test.shape[0], args.n_ensemble))
        score_fold_SB = np.empty((X_SB.shape[0], args.n_ensemble))

        # Store feature importances
        feature_importance_list = []

        for i_tree in range(args.n_ensemble):
            random_seed = ifold * args.n_ensemble + i_tree + 1


            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_full,
                y_train_full,
                test_size = 0.25,
                shuffle=True,
                random_state=random_seed
            ) # TODO: need more sophisticated splitting, but should be fine for now

            # Class weight balancing
            sample_weights = compute_sample_weight('balanced', y_train)
            sample_weights_valid = compute_sample_weight('balanced', y_valid)

            bst_i = XGBClassifier(
                n_estimators=random.choice([300, 500, 700]),
                max_depth=random.choice([3, 4, 5, 6]),
                learning_rate=np.random.uniform(0.03, 0.2),
                subsample=np.random.uniform(0.6, 1.0),
                min_child_weight=random.choice([1, 5, 10]),
                early_stopping_rounds=bdt_hyperparameters["early_stopping_rounds"],
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=random_seed
            )
            bst_i.fit(X_train,
                      y_train,
                      sample_weight=sample_weights,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      sample_weight_eval_set = [sample_weights, sample_weights_valid],
                      verbose=False,
            )
            results = bst_i.evals_result()
            losses = results['validation_0']['logloss']
            losses_valid = results['validation_1']['logloss']

            best_epoch = bst_i.best_iteration

            score_fold[:, i_tree] = bst_i.predict_proba(X_test, iteration_range=(0, best_epoch))[:, 1]
            score_fold_SB[:, i_tree] = bst_i.predict_proba(X_SB, iteration_range=(0, best_epoch))[:, 1]
            score_fold_no_signal[:, i_tree] = bst_i.predict_proba(X_no_signal, iteration_range=(0, best_epoch))[:, 1]

            # === NEW: Feature importance extraction ===
            booster = bst_i.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            feature_names_xgb = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in
                                                                                 range(X_train.shape[1])]
            feature_names = X.columns
            importance_series = pd.Series(
                {f: importance_dict.get(f_xgb, 0.0)
                for f, f_xgb in zip(feature_names, feature_names_xgb)}
            )
            feature_importance_list.append(importance_series)

        # === NEW: After loop, compute and plot average importance ===
        importance_df = pd.DataFrame(feature_importance_list)
        mean_importance = importance_df.fillna(0).mean().sort_values(ascending=False)

        print("✅ Average Feature Importances (by gain):")
        print(mean_importance)

        # Optional: plot top 20
        top_k = 20  # number of top features to display
        top_features = mean_importance.head(top_k)  # sort for horizontal bar plot
        #top_features = mean_importance.head(top_k)[::-1]

        plt.figure(figsize=(10, 8))

        ax = sns.barplot(
            x=top_features.values,
            y=top_features.index,
            palette=sns.color_palette("viridis", n_colors=top_k)
        )

        plt.title(f"Top {top_k} Feature Importances (Average Gain)", fontsize=14, fontweight='bold')
        plt.xlabel("Average Gain", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(plotdir, f'feature_importance_fold_{ifold}.png'))

        score_fold = np.mean(score_fold, axis=1)
        score_fold_SB = np.mean(score_fold_SB, axis=1)
        score_fold_no_signal = np.mean(score_fold_no_signal, axis=1)

        # === XGBoost ===
        xgb_probs = score_fold
        xgb_probs_SB = score_fold_SB
        df_SB_fold['xgb_prob'] = xgb_probs_SB
        df_no_signal_fold['xgb_prob'] = score_fold_no_signal
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
        roc_data['xgb'].append((fpr_xgb, tpr_xgb))
        auc_scores['xgb'].append(auc(fpr_xgb, tpr_xgb))
        fold_df = df.iloc[test_idx].copy()
        fold_df['xgb_prob'] =  xgb_probs
        print(f"✅ {ifold} Fold results: XGB AUC={auc_scores['xgb'][-1]:.2f}")

        model_list = ['xgb']

        # === AutoTabPFN ===
        pfn = AutoTabPFNClassifier(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ignore_pretraining_limits=True
        )

        if not(args.ignore is not None and args.ignore == "pfn"):
           pfn.fit(X_train_full, y_train_full)
           pfn_probs = pfn.predict_proba(X_test)[:, 1]
           pfn_probs_SB = pfn.predict_proba(X_SB)[:, 1]
           df_SB_fold['pfn_probs'] = pfn_probs_SB
           fpr_pfn, tpr_pfn, _ = roc_curve(y_test, pfn_probs)
           roc_data['pfn'].append((fpr_pfn, tpr_pfn))
           auc_scores['pfn'].append(auc(fpr_pfn, tpr_pfn))
           fold_df['pfn_probs'] = pfn_probs
           print(f"✅ {ifold} Fold results: PFN AUC={auc_scores['pfn'][-1]:.2f}")
           model_list.append('pfn')
        # === Store results for this fold ===
        test_results.append(fold_df)
        test_results_SB.append(df_SB_fold)
        test_results_no_signal.append(df_no_signal_fold)


    # Combine all test results
    results_df = pd.concat(test_results, ignore_index=True)
    results_df_SB = pd.concat(test_results_SB, ignore_index=True)
    results_df_no_signal = pd.concat(test_results_no_signal, ignore_index=True)

    print(results_df)

    # --- Plot ROC Curve ---
    plt.figure(figsize=(8, 6))
    for model in model_list:
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
    print(results_df_no_signal)
    results_df_no_signal.to_csv(os.path.join(outputdir, 'final_results_no_signal.csv'), index=False)
    print("✅ Results with no signal saved to final_results_no_signal.csv")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--region", type = str, default = "SR")
    parser.add_argument("--no_signal", action = "store_true", default = False)
    parser.add_argument("--knumber", type=int, default=3, help="Number of folds for K-Fold cross-validation")
    parser.add_argument("--n_ensemble", type=int, default=100, help="Number of ensembles for XGBoost")
    parser.add_argument("--filter_bad_event", action='store_true')
    parser.add_argument("--ignore", type=str, default = None)
    parser.add_argument("--drop", type=str, nargs = "+")
    parser.add_argument("--only_pc", action='store_true')
    parser.add_argument("--test_no_signal", action='store_true', default=False)

    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file

    train(args)

if __name__ == "__main__":
    main()

