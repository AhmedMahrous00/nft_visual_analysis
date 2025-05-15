import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import argparse
from datetime import datetime

SEED = 42
K_FOLDS = 10
OUTPUT_DIR = "outputs/traditional_simple"
os.makedirs(os.path.join(OUTPUT_DIR, "summaries"), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if hasattr(lgb, 'set_seed'): 
        lgb.set_seed(seed)

set_seed(SEED)

def get_visual_features():
    return [
        'mean_hue', 'mean_saturation', 'mean_value', 'colorfulness', 'brightness',
        'rule_of_thirds_score', 'horizontal_symmetry', 'vertical_symmetry',
        'texture_edge_density', 'entropy', 'contrast', 'sharpness',
        'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'num_objects', 'white_space_percentage', 'spatial_complexity',
        'simplicity_edge_density', 'color_simplicity', 'shape_simplicity', 'gradient_complexity',
        'lbp_hist_mean', 'lbp_hist_std', 'hog_mean', 'hog_std', 'hu_moment_1',
        'hu_moment_2', 'saliency_mean', 'saliency_std'
    ]

def get_simple_models():
    return {
        'ols': LinearRegression(),
        'ridge': Ridge(random_state=SEED),
        'knn': KNeighborsRegressor(n_jobs=-1),
        'xgb': xgb.XGBRegressor(
            random_state=SEED,
            n_jobs=-1,
            tree_method='hist' 
        ),
        'lgb': lgb.LGBMRegressor(
            random_state=SEED,
            n_jobs=-1,
            force_col_wise=True 
        )
    }

def evaluate_model(model, X_train, X_val, y_train, y_val):
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        trues = y_val
        
        r2_log = r2_score(trues, preds)
        rmse_log = np.sqrt(mean_squared_error(trues, preds))
        
        preds_o = np.expm1(preds)
        trues_o = np.expm1(trues)
        r2_o = r2_score(trues_o, preds_o)
        rmse_o = np.sqrt(mean_squared_error(trues_o, preds_o))
        
        return {
            'r2_log': float(r2_log), 'rmse_log': float(rmse_log),
            'r2_orig': float(r2_o), 'rmse_orig': float(rmse_o),
            'error': None
        }
    except Exception as e:
        return {
            'r2_log': None, 'rmse_log': None,
            'r2_orig': None, 'rmse_orig': None,
            'error': str(e)
        }

def run_evaluation_cv(df, model, feature_cols, target_col, cv_type="pooled", quiet=False):
    X = df[feature_cols]
    y = df[target_col]
    
    fold_metrics_list = []

    if cv_type == "pooled":
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        desc = f"Running {cv_type} CV for {model.__class__.__name__}"
        for fold, (tr_idx, va_idx) in tqdm(enumerate(kf.split(X), start=1), total=K_FOLDS, desc=desc, disable=quiet):
            X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
            y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
            metrics = evaluate_model(model, X_train, X_val, y_train, y_val)
            if metrics['error'] is None:
                fold_metrics_list.append(metrics)
    elif cv_type == "within":
        if 'collection_unique' not in df.columns:
            if not quiet: print("Error: 'collection_unique' column missing for within-collection CV.")
            return {'error': "'collection_unique' column missing.", 'success_rate': 0.0}
            
        unique_collection_names = df['collection_unique'].unique()
        for unique_name in tqdm(unique_collection_names, desc=f"Running {cv_type} CV for {model.__class__.__name__} (Collections)", disable=quiet):
            coll_df = df[df['collection_unique'] == unique_name]
            
            if len(coll_df) < K_FOLDS:
                if not quiet: print(f"  Skipping Collection '{unique_name}' (too few samples: {len(coll_df)})")
                continue
                
            X_coll, y_coll = coll_df[feature_cols], coll_df[target_col]
            kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
            coll_fold_metrics = []
            for fold, (tr_idx, va_idx) in tqdm(enumerate(kf.split(X_coll), start=1), total=K_FOLDS, desc=f"    Collection '{unique_name}' Folds", leave=False, disable=quiet):
                X_train, X_val = X_coll.iloc[tr_idx], X_coll.iloc[va_idx]
                y_train, y_val = y_coll.iloc[tr_idx], y_coll.iloc[va_idx]
                metrics = evaluate_model(model, X_train, X_val, y_train, y_val)
                if metrics['error'] is None:
                    coll_fold_metrics.append(metrics)

            if coll_fold_metrics:
                averaged_collection_metrics = {}
                if coll_fold_metrics[0]:
                    for key in coll_fold_metrics[0].keys():
                        if key != 'error':
                            valid_metric_values = [m[key] for m in coll_fold_metrics if m.get(key) is not None]
                            if valid_metric_values:
                                averaged_collection_metrics[key] = np.mean(valid_metric_values)

                fold_metrics_list.append({
                    'collection_name': unique_name,
                    'reported_name_for_metrics': unique_name,
                    'fold_metrics': coll_fold_metrics,
                    'average_metrics': averaged_collection_metrics
                })
            elif not quiet:
                 print(f"  Collection '{unique_name}': No successful folds out of {K_FOLDS} total folds.")

    elif cv_type == "cross":
        if 'collection_unique' not in df.columns:
            if not quiet: print("Error: 'collection_unique' column missing for cross-collection CV.")
            return {'error': "'collection_unique' column missing.", 'success_rate': 0.0}

        unique_collection_names = df['collection_unique'].unique()
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        desc = f"Running {cv_type} CV for {model.__class__.__name__}"
        for fold, (tr_coll_idx, va_coll_idx) in tqdm(enumerate(kf.split(unique_collection_names), start=1), total=K_FOLDS, desc=desc, disable=quiet):
            train_collection_names = unique_collection_names[tr_coll_idx]
            val_collection_names = unique_collection_names[va_coll_idx]
            
            X_train = df[df['collection_unique'].isin(train_collection_names)][feature_cols]
            y_train = df[df['collection_unique'].isin(train_collection_names)][target_col]
            X_val = df[df['collection_unique'].isin(val_collection_names)][feature_cols]
            y_val = df[df['collection_unique'].isin(val_collection_names)][target_col]
            
            if X_val.empty: 
                if not quiet: print(f"  Fold {fold}: Skipped, validation set empty for collections {val_collection_names}")
                continue
            metrics = evaluate_model(model, X_train, X_val, y_train, y_val)
            if metrics['error'] is None:
                fold_metrics_list.append(metrics)
    
    if not fold_metrics_list:
        return {'error': f'{cv_type} CV failed for all folds/collections.', 'success_rate': 0.0}

    avg_results = {}
    if fold_metrics_list: 
        if cv_type == "within":
            all_fold_metrics = []
            for coll_metrics in fold_metrics_list:
                if 'fold_metrics' in coll_metrics:
                    all_fold_metrics.extend(coll_metrics['fold_metrics'])
            
            keys_to_average = [k for k in all_fold_metrics[0].keys() if k not in ['error']]
            for key in keys_to_average:
                valid_values = [m[key] for m in all_fold_metrics if m.get(key) is not None]
                if valid_values:
                    avg_results[key + '_mean'] = np.mean(valid_values)
                    avg_results[key + '_std'] = np.std(valid_values)
        else:
            keys_to_average = [k for k in fold_metrics_list[0].keys() if k not in ['error']]
            for key in keys_to_average:
                valid_values = [m[key] for m in fold_metrics_list if isinstance(m, dict) and m.get(key) is not None]
                if valid_values:
                    avg_results[key + '_mean'] = np.mean(valid_values)
                    avg_results[key + '_std'] = np.std(valid_values)
    
    num_successful = len([m for m in fold_metrics_list if m.get('r2_log') is not None])
    total_possible_evals = K_FOLDS
    if cv_type == 'within':
        collection_counts = df['collection_unique'].value_counts()
        collections_meeting_min_size = collection_counts[collection_counts >= K_FOLDS].index
        total_possible_folds_within = len(collections_meeting_min_size) * K_FOLDS

        if total_possible_folds_within > 0:
            num_successful = sum(len(m_list) for m_list in fold_metrics_list if isinstance(m_list, list) and any(m.get('r2_log') for m in m_list))
            num_successful_collections = len(fold_metrics_list) 
            num_attempted_collections = len(collections_meeting_min_size)
            total_possible_evals = num_attempted_collections if num_attempted_collections > 0 else 1 
            num_successful = num_successful_collections
        else:
            total_possible_evals = 1 
            num_successful = 0

    avg_results['success_rate'] = num_successful / total_possible_evals if total_possible_evals > 0 else 0.0
    avg_results['num_successful_evals'] = num_successful
    avg_results['num_total_possible_evals'] = total_possible_evals

    if not quiet:
        r2_mean_val = avg_results.get('r2_log_mean', 'N/A')
        r2_std_val = avg_results.get('r2_log_std', 'N/A')
        if isinstance(r2_mean_val, float) and isinstance(r2_std_val, float):
            print(f"  Avg R2_log: {r2_mean_val:.3f} ± {r2_std_val:.3f}")
        else:
            print(f"  Avg R2_log: {r2_mean_val} ± {r2_std_val}")
    return {
        'summary_stats': avg_results,
        'detailed_fold_metrics': fold_metrics_list
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simplified Traditional ML Pipeline for NFT Price Prediction")
    parser.add_argument('--input_csv', type=str, default="outputs/traditional/summaries/nft_features_full_with_collection_unique.csv", help='Path to the input CSV file with collection_unique column')
    parser.add_argument('--min_collection_size', type=int, default=100, help='Minimum number of samples for a collection to be included')
    parser.add_argument('--test_run', action='store_true', help='Use 5% of data for quick testing')
    args = parser.parse_args()

    print("Starting simplified traditional ML pipeline...")

    print(f"\nLoading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded data with shape: {df.shape}")

    if args.test_run:
        print("\nTEST MODE: Using 5% of data")
        df = df.sample(frac=0.20, random_state=SEED)
        print(f"Sampled data shape: {df.shape}")

    if 'collection_unique' not in df.columns:
        print(f"Error: The required 'collection_unique' column is not found in {args.input_csv}.")
        print("Please ensure you are using a CSV processed by 'extract_collection_name.py' or that the column exists.")
        exit(1)
    
    print("\nFiltering collections based on 'collection_unique' column...")
    collection_counts = df['collection_unique'].value_counts()
    valid_collections = collection_counts[collection_counts >= args.min_collection_size].index
    df = df[df['collection_unique'].isin(valid_collections)].copy()
    print(f"Filtered to {len(valid_collections)} collections (from 'collection_unique') with at least {args.min_collection_size} samples.")
    print(f"Total samples after filtering: {len(df)}")
    
    if df.empty:
        print("DataFrame is empty after filtering. Exiting.")
        exit()
        
    feature_cols = get_visual_features()
    for col in feature_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
        elif col not in df.columns:
            print(f"Warning: Feature column '{col}' not found in DataFrame. Skipping NaN fill for it.")


    print("\n--- Step 1: Model Screening (K-Fold CV) ---")
    X_screen = df[feature_cols].copy()
    y_screen = df['log_price_clipped'].copy()
    
    models_to_screen = get_simple_models()
    screening_results = {}
    kf_screen = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for name, model_instance in tqdm(models_to_screen.items(), desc="Model Screening", leave=True):
        print(f"\n  Screening {name} with {K_FOLDS}-Fold CV...")
        model_fold_metrics = []
        for fold_num, (train_idx, val_idx) in tqdm(enumerate(kf_screen.split(X_screen, y_screen)), total=K_FOLDS, desc=f"    {name} Folds", leave=False):
            X_train_fold, X_val_fold = X_screen.iloc[train_idx], X_screen.iloc[val_idx]
            y_train_fold, y_val_fold = y_screen.iloc[train_idx], y_screen.iloc[val_idx]
            
            metrics = evaluate_model(model_instance, X_train_fold, X_val_fold, y_train_fold, y_val_fold)
            if metrics['error'] is None:
                model_fold_metrics.append(metrics)
            else:
                model_fold_metrics.append(metrics)

        current_model_r2_logs = [m['r2_log'] for m in model_fold_metrics if m.get('r2_log') is not None]
        mean_r2_log_for_model = np.mean(current_model_r2_logs) if current_model_r2_logs else None
        std_r2_log_for_model = np.std(current_model_r2_logs) if current_model_r2_logs and len(current_model_r2_logs) > 1 else None
        num_successful_folds = len(current_model_r2_logs)

        screening_results[name] = {
            'fold_metrics_list': model_fold_metrics,
            'mean_r2_log': mean_r2_log_for_model,
            'std_r2_log': std_r2_log_for_model,
            'successful_folds': num_successful_folds,
            'total_folds': K_FOLDS
        }

        if mean_r2_log_for_model is not None:
            if std_r2_log_for_model is not None:
                print(f"    {name} - Avg R2_log: {mean_r2_log_for_model:.3f} \u00b1 {std_r2_log_for_model:.3f} (from {num_successful_folds}/{K_FOLDS} folds)")
            else:
                print(f"    {name} - Avg R2_log: {mean_r2_log_for_model:.3f} (from {num_successful_folds}/{K_FOLDS} folds)")
        else:
            print(f"    {name} - R2_log: N/A (all folds failed or no R2_log produced)")

    valid_screened_models = [
        (name, res['mean_r2_log'])
        for name, res in screening_results.items()
        if res.get('mean_r2_log') is not None
    ]

    if not valid_screened_models:
        print("Error: All models failed during screening or returned no R2_log.")
        print("Screening results before exit:", screening_results)
        exit(1)
        
    best_model_name_screened = max(valid_screened_models, key=lambda x: x[1])[0]
    print(f"\nBest model from screening: {best_model_name_screened} (R2_log: {screening_results[best_model_name_screened]['mean_r2_log']:.3f})")

    print(f"\n--- Step 3: Final Evaluation for Best Screened Model ({best_model_name_screened}) ---")
    final_results_best_screened_model = {}
    
    model_for_final_eval = get_simple_models()[best_model_name_screened]
    
    print("\nEvaluating Pooled CV...")
    final_results_best_screened_model['pooled_cv'] = run_evaluation_cv(df, model_for_final_eval, feature_cols, 'log_price_clipped', "pooled")
    print("\nEvaluating Within-Collection CV...")
    final_results_best_screened_model['within_collection_cv'] = run_evaluation_cv(df, model_for_final_eval, feature_cols, 'log_price_clipped', "within")
    print("\nEvaluating Cross-Collection CV...")
    final_results_best_screened_model['cross_collection_cv'] = run_evaluation_cv(df, model_for_final_eval, feature_cols, 'log_price_clipped', "cross")

    all_pipeline_results = {
        'timestamp': TIMESTAMP,
        'settings': {
            'seed': SEED,
            'k_folds_final_eval': K_FOLDS,
            'min_collection_size': args.min_collection_size,
            'input_csv': args.input_csv
        },
        'screening_phase': {
            'results': screening_results,
            'best_model': best_model_name_screened,
            'best_model_r2_log': screening_results[best_model_name_screened].get('mean_r2_log')
        },
        'final_evaluation_best_screened_model': final_results_best_screened_model
    }

    results_filename = f'simple_traditional_pipeline_results_no_tuning_{TIMESTAMP}.json'
    results_path = os.path.join(OUTPUT_DIR, 'summaries', results_filename)
    with open(results_path, 'w') as f:
        json.dump(all_pipeline_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o))
    print(f"\nPipeline complete. All results saved to {results_path}")