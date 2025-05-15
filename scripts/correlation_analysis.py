import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr
import argparse
import random
from sklearn.decomposition import PCA 
from sklearn.model_selection import KFold
import statsmodels.api as sm_api
from statsmodels.stats.outliers_influence import variance_inflation_factor

# SETTINGS
SEED = 42
K_FOLDS_FOR_RC = 10 

# Reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if hasattr(lgb, 'set_seed'):
        lgb.set_seed(seed_value)

def get_visual_features():
    """Get list of all visual features extracted from images (copied from traditional_cv_nft_simple.py)"""
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

def get_lgbm_model(seed_value):
    """Get a LightGBM model instance"""
    return lgb.LGBMRegressor(
        random_state=seed_value,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1 
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NFT Collection Correlation Experiment")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file')

    args = parser.parse_args()

    set_seed(SEED)
    
    OUTPUT_DIR = "outputs/experiment_correlations"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, "collection_correlations.csv")

    print(f"Starting NFT collection correlation experiment...")
    print(f"Loading data from: {args.input_csv}")

    try:
        df = pd.read_csv(args.input_csv)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    if 'collection_unique' not in df.columns:
        print(f"Error: 'collection_unique' column not found in {args.input_csv}.")
        exit(1)
    
    if 'log_price_clipped' not in df.columns:
        print(f"Error: 'log_price_clipped' column (target) not found in {args.input_csv}.")
        exit(1)

    feature_cols = get_visual_features()
    target_col = 'log_price_clipped'

    # Check for missing feature columns and fill NaNs
    # Keep only feature columns that we want
    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(existing_feature_cols)
    if missing_cols:
        print(f"Warning: The following visual feature columns were not found in the CSV and will be ignored: {missing_cols}")
    
    if not existing_feature_cols:
        print("Error: No visual feature columns found in the CSV. Exiting.")
        exit(1)
        
    print(f"Using {len(existing_feature_cols)} available visual features for modeling.")

    for col in existing_feature_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled NaNs in '{col}' with median: {median_val}")

    unique_collections = df['collection_unique'].unique()
    print(f"Found {len(unique_collections)} unique collections.")

    model = get_lgbm_model(SEED)
    results_list = []

    for i, current_collection_name in enumerate(unique_collections):
        print(f"\nProcessing collection {i+1}/{len(unique_collections)}: {current_collection_name}")

        collection_df = df[df['collection_unique'] == current_collection_name].copy()

        if len(collection_df) < K_FOLDS_FOR_RC or len(collection_df) < 2:
            min_samples_needed = max(K_FOLDS_FOR_RC, 2)
            print(f"  Skipping collection '{current_collection_name}' due to insufficient samples (found {len(collection_df)}, need at least {min_samples_needed} for {K_FOLDS_FOR_RC}-fold CV and correlation).")
            results_list.append({
                'collection_unique': current_collection_name,
                'rc': np.nan,
                'fisher_z': np.nan,
                'num_test_samples': len(collection_df),
                'error': f'Insufficient samples for {K_FOLDS_FOR_RC}-fold CV and correlation'
            })
            continue
            
        X_collection = collection_df[existing_feature_cols]
        y_collection = collection_df[target_col]
        
        kf = KFold(n_splits=K_FOLDS_FOR_RC, shuffle=True, random_state=SEED)
        out_of_fold_predictions = pd.Series(index=X_collection.index, dtype=float)

        print(f"  Performing {K_FOLDS_FOR_RC}-fold CV within collection '{current_collection_name}' ({len(X_collection)} samples) to get predictions...")
        fold_successful = True
        for fold_num, (train_idx_iloc, val_idx_iloc) in enumerate(kf.split(X_collection, y_collection)):
            train_idx = X_collection.iloc[train_idx_iloc].index
            val_idx = X_collection.iloc[val_idx_iloc].index

            X_train_fold, X_val_fold = X_collection.loc[train_idx], X_collection.loc[val_idx]
            y_train_fold, y_val_fold = y_collection.loc[train_idx], y_collection.loc[val_idx]

            if X_train_fold.empty or X_val_fold.empty:
                print(f"    Fold {fold_num+1}: Skipping due to empty train/val split.")
                out_of_fold_predictions.loc[val_idx] = np.nan 
                fold_successful = False 
                continue
            
            try:
                model.fit(X_train_fold, y_train_fold)
                fold_preds = model.predict(X_val_fold)
                out_of_fold_predictions.loc[val_idx] = fold_preds
            except Exception as e:
                print(f"    Fold {fold_num+1}: Error during training/prediction: {e}")
                out_of_fold_predictions.loc[val_idx] = np.nan
                fold_successful = False

        # After all folds, calculate rc using the out_of_fold_predictions
        # Drop NaNs from predictions (if any folds failed) and corresponding true values
        valid_indices_for_rc = out_of_fold_predictions.notna() & y_collection.notna()
        y_collection_clean = y_collection[valid_indices_for_rc]
        oof_preds_clean = out_of_fold_predictions[valid_indices_for_rc]

        num_samples_for_rc = len(y_collection_clean)

        if num_samples_for_rc < 2:
            print(f"  Not enough valid (non-NaN) pairs ({num_samples_for_rc}) to calculate correlation for '{current_collection_name}\' after {K_FOLDS_FOR_RC}-fold CV.")
            rc_val = np.nan
            z_val = np.nan
            error_msg = 'Not enough valid pairs for correlation after K-fold CV' if fold_successful else 'Error during K-fold CV导致不足样本'
        else:
            rc_val, _ = pearsonr(y_collection_clean, oof_preds_clean)
            z_val = np.arctanh(rc_val)
            error_msg = None if fold_successful else 'Errors occurred in some CV folds'
            print(f"  Collection: {current_collection_name}, Samples for rc: {num_samples_for_rc}, Pearson rc: {rc_val:.4f}, Fisher's z: {z_val:.4f}")

        results_list.append({
            'collection_unique': current_collection_name,
            'rc': rc_val,
            'fisher_z': z_val,
            'num_test_samples': num_samples_for_rc, 
            'error': error_msg
        })
            
    results_df = pd.DataFrame(results_list)
    
    print(f"\nExperiment finished. Saving results to {output_filepath}")
    try:
        results_df.to_csv(output_filepath, index=False)
        print(f"Results saved successfully to {output_filepath}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("\n--- Starting Step 2: Aggregate Feature Calculation and PCA ---")

    
    print("Calculating aggregate features per collection...")
    agg_funcs = ['mean', 'median', 'var']
    
    # Aggregates for visual features
    collection_agg_features_list = []
    for col_name, group_df in df.groupby('collection_unique'):
        agg_data = {'collection_unique': col_name}
        for feat_col in existing_feature_cols:
            for func in agg_funcs:
                agg_data[f'{feat_col}_{func}'] = group_df[feat_col].agg(func)
        # Aggregates for log_price_clipped
        for func in agg_funcs:
            agg_data[f'{target_col}_{func}'] = group_df[target_col].agg(func)
        collection_agg_features_list.append(agg_data)
    
    collection_features_df = pd.DataFrame(collection_agg_features_list)
    print(f"Calculated aggregate features for {len(collection_features_df)} collections.")

    # Log-transform the visual feature aggregates (all except log_price_clipped. it is already transformed)
    print("Log-transforming aggregate visual features (using np.log1p)...")
    transformed_feature_names = []
    for col in collection_features_df.columns:
        if col != 'collection_unique' and not col.startswith(target_col):
            collection_features_df[col] = pd.to_numeric(collection_features_df[col], errors='coerce')
            collection_features_df[col] = np.log1p(collection_features_df[col])
            transformed_feature_names.append(col)
    print(f"Log-transformed {len(transformed_feature_names)} aggregate visual feature columns.")

    # Merge with Fisher's z scores
    final_df_for_regression = pd.merge(results_df[['collection_unique', 'fisher_z', 'rc']], 
                                       collection_features_df, 
                                       on='collection_unique',
                                       how='inner') 
    
    final_df_for_regression.dropna(subset=['fisher_z'], inplace=True)
    final_df_for_regression = final_df_for_regression[~np.isinf(final_df_for_regression['fisher_z'])]

    print(f"Merged Fisher's z with aggregate features. Resulting shape for PCA/regression: {final_df_for_regression.shape}")
    
    if final_df_for_regression.empty:
        print("Error: No data remaining after merging and cleaning for PCA/regression. Exiting further analysis.")
        exit(1)
    if len(final_df_for_regression) < 2:
        print("Error: Less than 2 samples remaining for PCA/regression. Exiting further analysis.")
        exit(1)
        
    pca_input_features = [col for col in final_df_for_regression.columns 
                          if col not in ['collection_unique', 'fisher_z', 'rc'] and not col.startswith(target_col)]
    
    if not pca_input_features:
        print("Error: No aggregate visual features available for PCA. Exiting further analysis.")
        exit(1)

    X_pca = final_df_for_regression[pca_input_features]
    
    if X_pca.isnull().any().any():
        print("Warning: NaNs found in PCA input features. Filling with column means.")
        X_pca = X_pca.fillna(X_pca.mean())
    
    if X_pca.empty or X_pca.shape[1] == 0:
        print("Error: PCA input data is empty or has no features after processing. Exiting PCA.")
    else:
        print(f"Running PCA on {X_pca.shape[1]} aggregate visual features for {X_pca.shape[0]} collections...")
        n_components_pca = min(5, X_pca.shape[0], X_pca.shape[1])
        
        if n_components_pca > 0:
            pca = PCA(n_components=n_components_pca, random_state=SEED)
            try:
                pca.fit(X_pca)
                explained_variance_ratio = pca.explained_variance_ratio_
                print("Explained variance by top components:")
                for i, ratio in enumerate(explained_variance_ratio):
                    print(f"  PC{i+1}: {ratio:.4f} (Cumulative: {np.sum(explained_variance_ratio[:i+1]):.4f})")

                print("\nTop contributing features for each Principal Component:")
                num_top_features_to_show = 5 # You can adjust this number
                for i in range(n_components_pca):
                    print(f"  --- PC{i+1} (Explains {explained_variance_ratio[i]:.2%} variance) ---")
                    loadings = pca.components_[i]
                    loadings_series = pd.Series(loadings, index=X_pca.columns)
                    sorted_loadings = loadings_series.abs().sort_values(ascending=False)
                    top_n_features = loadings_series[sorted_loadings.head(num_top_features_to_show).index]
                    for feature_name, loading_value in top_n_features.items():
                        print(f"    {feature_name}: {loading_value:.4f}")
            except Exception as e:
                print(f"Error during PCA fitting: {e}")
        else:
            print("Not enough samples or features to run PCA with n_components > 0.")

    print("\n--- Starting Step 3: Aggregate Feature Correlation and VIF Calculation ---")
    if not X_pca.empty and X_pca.shape[1] > 1:
        print("Calculating correlation matrix for aggregate visual features...")
        try:
            correlation_matrix = X_pca.corr()
            corr_matrix_filename = "aggregate_features_correlation_matrix.csv"
            corr_matrix_filepath = os.path.join(OUTPUT_DIR, corr_matrix_filename)
            correlation_matrix.to_csv(corr_matrix_filepath)
            print(f"Correlation matrix saved to {corr_matrix_filepath}")
        except Exception as e:
            print(f"Error calculating/saving correlation matrix: {e}")

        print("Calculating VIF scores for aggregate visual features...")
        try:
            X_pca_with_const = sm_api.add_constant(X_pca, prepend=False) 
            vif_data = []
            for i, col_name in enumerate(X_pca.columns):
                try:
                    vif_value = variance_inflation_factor(X_pca_with_const.values, X_pca_with_const.columns.get_loc(col_name))
                    vif_data.append({'feature': col_name, 'vif': vif_value})
                except Exception as vif_e:
                    print(f"  Could not calculate VIF for {col_name}: {vif_e}. Setting to NaN.")
                    vif_data.append({'feature': col_name, 'vif': np.nan}) 
            
            vif_df = pd.DataFrame(vif_data)
            vif_df.sort_values(by='vif', ascending=False, inplace=True)
            vif_scores_filename = "aggregate_features_vif_scores.csv"
            vif_scores_filepath = os.path.join(OUTPUT_DIR, vif_scores_filename)
            vif_df.to_csv(vif_scores_filepath, index=False)
            print(f"VIF scores saved to {vif_scores_filepath}")
            print("\nTop 10 features by VIF score:")
            print(vif_df.head(10).to_string())

        except Exception as e:
            print(f"Error calculating/saving VIF scores: {e}")
    else:
        print("Skipping correlation matrix and VIF calculation as there are not enough features or data in X_pca.")

    regression_data_filename = "collection_regression_data_with_fisher_z.csv"
    regression_data_filepath = os.path.join(OUTPUT_DIR, regression_data_filename)
    try:
        final_df_for_regression.to_csv(regression_data_filepath, index=False)
        print(f"Data prepared for regression (including Fisher's z and aggregate features) saved to {regression_data_filepath}")
    except Exception as e:
        print(f"Error saving regression data CSV: {e}")

    print("\nScript finished.") 