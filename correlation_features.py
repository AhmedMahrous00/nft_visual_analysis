import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.utils import resample
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.colors import Normalize
from matplotlib.cm import viridis


N_BOOTSTRAPS = 1000
ELASTIC_NET_L1_RATIO = 0.5
TOP_N_FEATURES = 6
RANDOM_STATE = 42


FEATURE1_NAME = "log_price_clipped_var"
FEATURE2_NAME = "contrast_var"
TARGET_NAME = "fisher_z"

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def filter_candidate_features(correlations_csv_path, output_csv_path):
    print(f"\n--- Step 1: Filtering Candidate Features ---")
    try:
        correlations_df = pd.read_csv(correlations_csv_path)
    except FileNotFoundError:
        print(f"Error: Correlations file not found at {correlations_csv_path}")
        return None, None


    candidates = correlations_df[
        (correlations_df['abs_spearman_correlation'] > 0.3) &
        (correlations_df['abs_pearson_correlation'] > 0.3) &
        ((correlations_df['pearson_p_value'] < 0.1) | (correlations_df['spearman_p_value'] < 0.1))
    ]

    print(f"Found {len(candidates)} candidate features based on correlation and p-value thresholds.")
    
    if candidates.empty:
        print("No candidate features found. Exiting further steps.")
        return None, None

    try:
        candidates.to_csv(output_csv_path, index=False)
        print(f"Candidate features and their correlations saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving candidate features CSV: {e}")
        

    candidate_feature_names = candidates['feature'].tolist()
    print("Candidate features selected:")
    for fname in candidate_feature_names:
        print(f"  - {fname}")
    return candidates, candidate_feature_names


def bootstrap_elastic_net_selection(main_data_csv_path, candidate_feature_names, output_csv_path):
    print(f"\n--- Step 2: Bootstrap Elastic Net Feature Selection ---")
    try:
        df_main = pd.read_csv(main_data_csv_path)
    except FileNotFoundError:
        print(f"Error: Main data file not found at {main_data_csv_path}")
        return None, None

    if 'fisher_z' not in df_main.columns:
        print("Error: 'fisher_z' column not found in main data file.")
        return None, None

    missing_candidates = [f for f in candidate_feature_names if f not in df_main.columns]
    if missing_candidates:
        print(f"Error: The following candidate features are missing from the main data file: {missing_candidates}")
        return None, None


    X_full = df_main[candidate_feature_names].copy()
    y_full = df_main['fisher_z'].copy()


    if y_full.isnull().any():
        not_nan_y_indices = y_full.notna()
        y_full = y_full[not_nan_y_indices]
        X_full = X_full[not_nan_y_indices]
        print(f"Dropped {sum(~not_nan_y_indices)} rows due to NaNs in 'fisher_z'. Shape after: y({y_full.shape}), X({X_full.shape})")
    
    if X_full.empty or y_full.empty:
        print("Error: Data is empty after handling NaNs in 'fisher_z'.")
        return None, None


    if X_full.isnull().any().any():
        print("Warning: NaNs found in candidate features (X). Filling with column means.")
        for col in X_full.columns:
            if X_full[col].isnull().any():
                X_full[col].fillna(X_full[col].mean(), inplace=True)
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled, columns=candidate_feature_names, index=X_full.index)

    feature_selection_counts = pd.Series(0, index=candidate_feature_names)
    n_samples = X_scaled_df.shape[0]

    if n_samples < 2:
        print("Error: Not enough samples (<2) to perform bootstrapping after NaN handling.")
        return None, None
    if n_samples < 3 and isinstance(LeaveOneOut(), LeaveOneOut):
         print(f"Warning: Number of samples ({n_samples}) is very small for LeaveOneOut CV. Consider a different CV strategy if ElasticNetCV fails or gives poor results.")

    print(f"Starting {N_BOOTSTRAPS} bootstrap iterations with {n_samples} samples...")
    for i in range(N_BOOTSTRAPS):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{N_BOOTSTRAPS}")


        
        boot_indices = resample(X_scaled_df.index, n_samples=n_samples, replace=True, random_state=RANDOM_STATE + i)
        
        X_boot = X_scaled_df.loc[boot_indices]
        y_boot = y_full.loc[boot_indices]

        if X_boot.empty or len(X_boot) < 2: 
            
            continue
        

        if y_boot.nunique() < 2:
            
            continue

        try:
            
            
            elastic_cv = ElasticNetCV(
                l1_ratio=ELASTIC_NET_L1_RATIO, 
                cv=LeaveOneOut(), 
                random_state=RANDOM_STATE + i, 
                n_jobs=-1, 
                max_iter=10000, 
                alphas=None, 
                tol=1e-4 
            )
            elastic_cv.fit(X_boot, y_boot)


            selected_coeffs_mask = elastic_cv.coef_ != 0 
            
            

            candidate_feature_names_np = np.array(candidate_feature_names)
            
            selected_names_this_iteration = candidate_feature_names_np[selected_coeffs_mask]
            

            feature_selection_counts.loc[selected_names_this_iteration] += 1
        except Exception as e:
            print(f"  Error in bootstrap iteration {i+1} with ElasticNetCV: {e}. Skipping iteration.")
            continue


    counts_df = feature_selection_counts.reset_index()
    counts_df.columns = ['feature', 'selection_count']
    counts_df.sort_values(by='selection_count', ascending=False, inplace=True)

    try:
        counts_df.to_csv(output_csv_path, index=False)
        print(f"Bootstrap feature selection counts saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving bootstrap selection counts CSV: {e}")

    top_n_from_bootstrap = counts_df.head(TOP_N_FEATURES)['feature'].tolist()
    print(f"\nTop {len(top_n_from_bootstrap)} features from bootstrap (target {TOP_N_FEATURES}):")
    for fname in top_n_from_bootstrap:
        count = counts_df[counts_df['feature'] == fname]['selection_count'].iloc[0]
        print(f"  - {fname} (Selected in {count}/{N_BOOTSTRAPS} bootstraps)")
        
    if not top_n_from_bootstrap:
        print("Warning: No features were consistently selected by bootstrap Elastic Net.")
        

    return counts_df, top_n_from_bootstrap

def analyze_top_features(main_data_csv_path, top_n_feature_names, output_csv_path):
    print(f"\n--- Step 3: Analyze Top {len(top_n_feature_names)} Features (Correlation Matrix) ---")
    if not top_n_feature_names:
        print("No top features provided to analyze. Skipping step 3.")
        return

    try:
        df_main = pd.read_csv(main_data_csv_path)
    except FileNotFoundError:
        print(f"Error: Main data file not found at {main_data_csv_path}")
        return

    missing_top_features = [f for f in top_n_feature_names if f not in df_main.columns]
    if missing_top_features:
        print(f"Error: The following top features are missing from the main data file: {missing_top_features}")
        return

    X_top_features = df_main[top_n_feature_names].copy()


    if X_top_features.isnull().any().any():
        print("Warning: NaNs found in top features. Filling with column means before correlation.")
        for col in X_top_features.columns:
            if X_top_features[col].isnull().any():
                X_top_features[col].fillna(X_top_features[col].mean(), inplace=True)
    
    if X_top_features.empty or X_top_features.shape[1] < 2:
        print("Not enough data or features (<2) among top selected features to compute a correlation matrix. Skipping.")
        return

    try:
        correlation_matrix_top_n = X_top_features.corr()
        print("\nCorrelation Matrix of Top Features:")
        print(correlation_matrix_top_n)
        
        correlation_matrix_top_n.to_csv(output_csv_path)
        print(f"Correlation matrix for top {len(top_n_feature_names)} features saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error calculating/saving correlation matrix for top features: {e}")


def fit_ols_on_top_features(main_data_csv_path, top_n_feature_names, summary_path, plot_path):
    print(f"\n--- Step 4: OLS Regression on Top {len(top_n_feature_names)} Features ---")
    if not top_n_feature_names:
        print("No top features provided for OLS regression. Skipping step 4.")
        return

    try:
        df_main = pd.read_csv(main_data_csv_path)
    except FileNotFoundError:
        print(f"Error: Main data file not found at {main_data_csv_path}")
        return

    if 'fisher_z' not in df_main.columns:
        print("Error: 'fisher_z' column not found in main data file.")
        return

    missing_top_features = [f for f in top_n_feature_names if f not in df_main.columns]
    if missing_top_features:
        print(f"Error: The following top features are missing from the main data file for OLS: {missing_top_features}")
        return

    X_ols = df_main[top_n_feature_names].copy()
    y_ols = df_main['fisher_z'].copy()


    if y_ols.isnull().any():
        not_nan_y_indices = y_ols.notna()
        y_ols = y_ols[not_nan_y_indices].reset_index(drop=True) 
        X_ols = X_ols[not_nan_y_indices].reset_index(drop=True) 
        print(f"Dropped {sum(~not_nan_y_indices)} rows due to NaNs in 'fisher_z' for OLS. Shape after: y({y_ols.shape}), X({X_ols.shape})")

    if X_ols.empty or y_ols.empty or X_ols.shape[0] < 2:
        print("Error: Data for OLS is empty or has too few samples (<2) after NaN handling.")
        return


    if X_ols.isnull().any().any():
        print("Warning: NaNs found in OLS features (X). Filling with column means.")
        for col in X_ols.columns:
            if X_ols[col].isnull().any():
                X_ols[col].fillna(X_ols[col].mean(), inplace=True)
    

    X_ols_with_const = sm.add_constant(X_ols)

    try:
        ols_model = sm.OLS(y_ols, X_ols_with_const).fit()
        print("\nOLS Regression Summary:")
        print(ols_model.summary())


        with open(summary_path, 'w') as f:
            f.write(ols_model.summary().as_text())
        print(f"OLS summary saved to: {summary_path}")


        predictions = ols_model.predict(X_ols_with_const)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_ols, predictions, alpha=0.7, label='Actual vs. Predicted')
        min_val = min(y_ols.min(), predictions.min())
        max_val = max(y_ols.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
        plt.xlabel("Actual Fisher's z")
        plt.ylabel("Predicted Fisher's z")
        plt.title(f"OLS: Actual vs. Predicted Fisher's z (Top {len(top_n_feature_names)} Features)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"OLS predicted vs. actual plot saved to: {plot_path}")
        plt.close() 

    except Exception as e:
        print(f"Error during OLS fitting or plotting: {e}")


def fit_rf_and_permutation_importance(main_data_csv_path, top_n_feature_names, output_csv_path):
    print(f"\n--- Step 5: Random Forest and Permutation Importance (Top {len(top_n_feature_names)} Features) ---")
    if not top_n_feature_names:
        print("No top features provided for Random Forest. Skipping step 5.")
        return None 

    try:
        df_main = pd.read_csv(main_data_csv_path)
    except FileNotFoundError:
        print(f"Error: Main data file not found at {main_data_csv_path}")
        return None

    if TARGET_NAME not in df_main.columns: 
        print(f"Error: '{TARGET_NAME}' column not found in main data file.")
        return None

    missing_top_features = [f for f in top_n_feature_names if f not in df_main.columns]
    if missing_top_features:
        print(f"Error: The following top features are missing from the main data file for RF: {missing_top_features}")
        return None

    X_rf = df_main[top_n_feature_names].copy()
    y_rf = df_main[TARGET_NAME].copy()


    if y_rf.isnull().any():
        not_nan_y_indices = y_rf.notna()
        y_rf = y_rf[not_nan_y_indices].reset_index(drop=True)
        X_rf = X_rf[not_nan_y_indices].reset_index(drop=True)
        print(f"Dropped {sum(~not_nan_y_indices)} rows due to NaNs in '{TARGET_NAME}' for RF. Shape after: y({y_rf.shape}), X({X_rf.shape})")

    if X_rf.empty or y_rf.empty or X_rf.shape[0] < 2:
        print("Error: Data for RF is empty or has too few samples (<2) after NaN handling.")
        return None


    if X_rf.isnull().any().any():
        print("Warning: NaNs found in RF features (X). Filling with column means.")
        for col in X_rf.columns:
            if X_rf[col].isnull().any():
                X_rf[col].fillna(X_rf[col].mean(), inplace=True)

    rf_model_main_fit = None 
    try:
        rf_model_main_fit = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, oob_score=True)
        rf_model_main_fit.fit(X_rf, y_rf)
        print("Random Forest model fitted.")


        r2_train = rf_model_main_fit.score(X_rf, y_rf)
        print(f"  1. In-sample R2 score: {r2_train:.4f}")


        r2_oob = rf_model_main_fit.oob_score_ if hasattr(rf_model_main_fit, 'oob_score_') else np.nan
        print(f"  2. Out-of-Bag (OOB) R2 score: {r2_oob:.4f}")


        rf_for_cv = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)


        num_samples_for_5_fold = len(X_rf)
        actual_cv_folds = 5
        if num_samples_for_5_fold < actual_cv_folds:
            r2_5fold_mean = np.nan
            r2_5fold_std = np.nan
            print(f"  3. {actual_cv_folds}-fold R2 score: N/A (not enough samples: {num_samples_for_5_fold} < {actual_cv_folds})")
        else:
            r2_5fold_scores = cross_val_score(rf_for_cv, X_rf, y_rf, cv=actual_cv_folds, scoring='r2', n_jobs=-1)
            r2_5fold_mean = r2_5fold_scores.mean()
            r2_5fold_std = r2_5fold_scores.std()
            print(f"  3. {actual_cv_folds}-fold R2 score: {r2_5fold_mean:.4f} \u00B1 {r2_5fold_std:.4f} (from {len(r2_5fold_scores)} folds)")
        

        r2_file_path = os.path.join(OUTPUT_SUBDIR, f"rf_r2_scores_top{len(top_n_feature_names)}.txt")
        with open(r2_file_path, 'w') as f:
            f.write(f"Random Forest R2 Scores (Top {len(top_n_feature_names)} features):\n")
            f.write(f"  1. In-sample R2: {r2_train:.4f}\n")
            f.write(f"  2. Out-of-Bag (OOB) R2: {r2_oob:.4f}\n")
            f.write(f"  3. {actual_cv_folds}-fold R2: {r2_5fold_mean:.4f} \u00B1 {r2_5fold_std:.4f}\n")
        print(f"Random Forest R2 scores saved to: {r2_file_path}")


        print("Calculating permutation importance...")
        perm_importance_result = permutation_importance(
            rf_model_main_fit, X_rf, y_rf, 
            n_repeats=30, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': X_rf.columns,
            'importance_mean': perm_importance_result.importances_mean,
            'importance_std': perm_importance_result.importances_std
        })
        perm_importance_df.sort_values(by='importance_mean', ascending=False, inplace=True)

        print("\nRandom Forest Permutation Importance:")
        print(perm_importance_df.to_string())


        perm_importance_df.to_csv(output_csv_path, index=False)
        print(f"Random Forest permutation importance saved to: {output_csv_path}")
        return rf_model_main_fit, X_rf 

    except Exception as e:
        print(f"Error during Random Forest fitting or permutation importance calculation: {e}")
        return rf_model_main_fit, X_rf 


def plot_colored_scatter_with_contours(df, f1_col, f2_col, target_col, output_dir, filename="scatter_contours.png"):
    """1. Colored 2-D scatter with regression contours."""
    plt.figure(figsize=(10, 8))
    plot_df = df[[f1_col, f2_col, target_col]].dropna()
    if plot_df.empty or len(plot_df) < 3: 
        print(f"Skipping {filename}: Not enough data after dropping NaNs for {f1_col}, {f2_col}, {target_col} (need >= 3).")
        plt.close()
        return

    x = plot_df[f1_col]
    y = plot_df[f2_col]
    z = plot_df[target_col]

    scatter = plt.scatter(x, y, c=z, cmap='viridis', edgecolors='k', alpha=0.7, s=60)
    plt.colorbar(scatter, label=target_col)

    try:
        plt.tricontour(x, y, z, levels=10, colors='k', alpha=0.6, linewidths=0.8)
    except Exception as e:
        print(f"Could not generate contours for {filename}: {e}")

    plt.xlabel(f1_col)
    plt.ylabel(f2_col)
    plt.title(f'Scatter: {f1_col} vs {f2_col} ({target_col} color & contours)')
    plt.grid(True, alpha=0.3)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Plot 1 (Scatter with Contours) saved to {filepath}")
    plt.close()

def plot_binned_heatmap(df, f1_col, f2_col, target_col, output_dir, filename="binned_heatmap.png"):
    """2. Binned heatmap / hexbin."""
    plt.figure(figsize=(10, 8))
    plot_df = df[[f1_col, f2_col, target_col]].dropna()
    if plot_df.empty or len(plot_df) < 3: 
        print(f"Skipping {filename}: Not enough data after dropping NaNs (need at least 3 for hexbin).")
        plt.close()
        return

    x = plot_df[f1_col]
    y = plot_df[f2_col]
    z = plot_df[target_col]
    try:
        hb = plt.hexbin(x, y, C=z, gridsize=20, cmap='viridis', reduce_C_function=np.mean, mincnt=1)
        plt.colorbar(hb, label=f'Mean {target_col}')
    except Exception as e:
        print(f"Could not generate hexbin plot for {filename}: {e}")
        plt.close()
        return
    
    plt.xlabel(f1_col)
    plt.ylabel(f2_col)
    plt.title(f'Hexbin: {f1_col} vs {f2_col} (Mean {target_col} in bin)')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Plot 2 (Binned Heatmap) saved to {filepath}")
    plt.close()

def plot_marginal_scatter_with_trends(df, f1_col, f2_col, target_col, output_dir, filename="marginal_scatter_trends.png"):
    """3. Marginal plots + scatter (customized)."""
    plot_df = df[[f1_col, f2_col, target_col]].dropna()
    if plot_df.empty or len(plot_df) < 5: 
        print(f"Skipping {filename}: Not enough data for marginal scatter with LOESS (need >= 5).")
        return

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), 
                          left=0.1, right=0.9, bottom=0.1, top=0.9, 
                          wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs[1,0])
    ax_hist_x = fig.add_subplot(gs[0,0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1,1], sharey=ax_scatter)
    ax_colorbar = fig.add_axes([0.92, 0.1, 0.02, 0.6]) 


    norm = Normalize(vmin=plot_df[target_col].min(), vmax=plot_df[target_col].max())
    cmap = viridis
    scatter_plot = ax_scatter.scatter(plot_df[f1_col], plot_df[f2_col], c=plot_df[target_col], cmap=cmap, norm=norm, s=60, edgecolors='w', alpha=0.8)
    ax_scatter.set_xlabel(f1_col)
    ax_scatter.set_ylabel(f2_col)


    sns.scatterplot(x=plot_df[f1_col], y=plot_df[target_col], ax=ax_hist_x, color=cmap(0.2), s=40, ec='w', alpha=0.6)
    sns.regplot(x=plot_df[f1_col], y=plot_df[target_col], ax=ax_hist_x, scatter=False, lowess=True, color='red', line_kws={'lw': 2.5})
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_x.set_ylabel(target_col)


    sns.scatterplot(y=plot_df[f2_col], x=plot_df[target_col], ax=ax_hist_y, color=cmap(0.8), s=40, ec='w', alpha=0.6)
    sns.regplot(y=plot_df[f2_col], x=plot_df[target_col], ax=ax_hist_y, scatter=False, lowess=True, color='red', line_kws={'lw': 2.5})
    ax_hist_y.tick_params(axis="y", labelleft=False)
    ax_hist_y.set_xlabel(target_col)
    
    fig.colorbar(scatter_plot, cax=ax_colorbar, label=target_col)
    fig.suptitle(f'Scatter of {f1_col} vs {f2_col} with Marginal Trends vs {target_col}', fontsize=16, y=0.95)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Plot 3 (Marginal Scatter with Trends) saved to {filepath}")
    plt.close(fig)

def plot_facetted_slices(df, main_effect_col, conditioning_col, target_col, output_dir, filename_prefix="facetted_slices"):
    """4. Facetted slices of a surface."""
    plot_df = df[[main_effect_col, conditioning_col, target_col]].dropna()
    if plot_df.empty or len(plot_df) < 15: 
        print(f"Skipping facetted slices for {main_effect_col} (conditioned on {conditioning_col}): Not enough data (need >= 15).")
        return

    try:
        plot_df['conditioning_bins'] = pd.qcut(plot_df[conditioning_col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    except ValueError:
        try:
            plot_df['conditioning_bins'] = pd.qcut(plot_df[conditioning_col], q=2, labels=['Low', 'High'], duplicates='drop')
        except ValueError:
            print(f"Could not create bins for {conditioning_col}. Skipping facetted plot.")
            return
    
    n_bins = plot_df['conditioning_bins'].nunique()
    if n_bins < 2:
        print(f"Skipping facetted slices: only {n_bins} unique bins created for {conditioning_col}.")
        return

    g = sns.FacetGrid(plot_df, col='conditioning_bins', height=4, aspect=1.1, col_wrap=min(n_bins, 3)) 
    g.map_dataframe(sns.scatterplot, x=main_effect_col, y=target_col, alpha=0.7, s=50, ec='w')
    
    for ax_idx, ax_data_tuple in enumerate(g.facet_data()):
        ax = g.axes.flat[ax_idx]
        
        
        
        current_data = ax_data_tuple[3]

        if len(current_data) >= 5 : 
             sns.regplot(x=current_data[main_effect_col], y=current_data[target_col],
                        ax=ax, scatter=False, lowess=True, color='red', line_kws={'lw':2})

    g.set_axis_labels(main_effect_col, target_col)
    g.set_titles(col_template=f"{conditioning_col}: {{col_name}}")
    title = f'Facetted: {main_effect_col} vs. {target_col} (Conditioned on {conditioning_col})'
    g.fig.suptitle(title, y=1.03)
    filepath = os.path.join(output_dir, f"{filename_prefix}_{main_effect_col}_by_{conditioning_col}.png")
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Plot 4 (Facetted Slices for {main_effect_col}) saved to {filepath}")
    plt.close()

def plot_partial_dependence_surface(rf_model, X_train_df, f1_name, f2_name, target_name_for_label, output_dir, filename="pdp_surface.png"):
    """5. Partial-dependence surface from Random Forest."""
    if rf_model is None:
        print("Skipping PDP surface: RF model is not available.")
        return
    if not (f1_name in X_train_df.columns and f2_name in X_train_df.columns):
        print(f"Skipping PDP: One or both features ('{f1_name}', '{f2_name}') not in RF model training columns: {X_train_df.columns.tolist()}")
        return
    if X_train_df.empty:
        print(f"Skipping PDP: X_train_df is empty.")
        return

    print(f"Generating PDP for features: '{f1_name}' and '{f2_name}'")
    try:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))
        PartialDependenceDisplay.from_estimator(
            rf_model,
            X_train_df, 
            features=[(f1_name, f2_name)], 
            ax=ax,
            grid_resolution=20 
        )
        ax.set_xlabel(f1_name)
        ax.set_ylabel(f2_name)
        ax.set_zlabel(f'Partial Dependence ({target_name_for_label})')
        plt.title(f'PDP Surface: {f1_name} & {f2_name} on {target_name_for_label}')
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Plot 5 (PDP Surface) saved to {filepath}")
        plt.close()
    except Exception as e:
        print(f"Error generating PDP surface plot for '{f1_name}' & '{f2_name}': {e}")
        plt.close()



def main():
    parser = argparse.ArgumentParser(description="Advanced feature selection and modeling pipeline.")
    parser.add_argument(
        '--correlations_input_csv', 
        type=str, 
        required=True,
        help='Path to the input CSV file containing feature correlations with Fisher\'s z.'
    )
    parser.add_argument(
        '--main_data_input_csv',
        type=str,
        required=True,
        help='Path to the main data CSV with Fisher\'s z and all aggregate features.'
    )

    args = parser.parse_args()


    OUTPUT_SUBDIR = "outputs/advanced_modeling"
    ensure_dir(OUTPUT_SUBDIR)

    candidate_features_df, candidate_feature_names = filter_candidate_features(
        args.correlations_input_csv,
        os.path.join(OUTPUT_SUBDIR, "candidate_features_correlations.csv")
    )
    if not candidate_feature_names:
        return

    bootstrap_selection_counts, top_n_from_bootstrap = bootstrap_elastic_net_selection(
        args.main_data_input_csv,
        candidate_feature_names,
        os.path.join(OUTPUT_SUBDIR, "bootstrap_feature_selection_counts.csv")
    )
    
    analyze_top_features(
        args.main_data_input_csv, 
        top_n_from_bootstrap, 
        os.path.join(OUTPUT_SUBDIR, f"top{TOP_N_FEATURES}_features_correlation_matrix.csv")
    )

    fit_ols_on_top_features(
        args.main_data_input_csv, 
        top_n_from_bootstrap, 
        os.path.join(OUTPUT_SUBDIR, f"ols_summary_top{TOP_N_FEATURES}.txt"),
        os.path.join(OUTPUT_SUBDIR, f"ols_pred_vs_actual_top{TOP_N_FEATURES}.png")
    )

    rf_model, X_rf_for_pdp = fit_rf_and_permutation_importance(
        args.main_data_input_csv, 
        top_n_from_bootstrap, 
        os.path.join(OUTPUT_SUBDIR, f"rf_permutation_importance_top{TOP_N_FEATURES}.csv")
    )


    print("\n--- Step 6: Generating Interaction Visualizations ---")
    try:
        df_for_plots = pd.read_csv(args.main_data_input_csv)
    except FileNotFoundError:
        print(f"Error: Main data file for plotting not found at {args.main_data_input_csv}. Skipping interaction plots.")
        df_for_plots = None
    
    if df_for_plots is not None:
        required_plot_cols = [FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME]
        missing_plot_cols = [col for col in required_plot_cols if col not in df_for_plots.columns]
        if missing_plot_cols:
            print(f"Warning: One or more specified columns for interaction plots are missing from {args.main_data_input_csv}: {missing_plot_cols}. Skipping these plots.")
        else:
            plot_colored_scatter_with_contours(df_for_plots, FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME, OUTPUT_SUBDIR)
            plot_binned_heatmap(df_for_plots, FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME, OUTPUT_SUBDIR)
            plot_marginal_scatter_with_trends(df_for_plots, FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME, OUTPUT_SUBDIR)
            plot_facetted_slices(df_for_plots, FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME, OUTPUT_SUBDIR, filename_prefix=f"facetted_slices")
            plot_facetted_slices(df_for_plots, FEATURE2_NAME, FEATURE1_NAME, TARGET_NAME, OUTPUT_SUBDIR, filename_prefix=f"facetted_slices")
            if rf_model is not None and X_rf_for_pdp is not None:
                plot_partial_dependence_surface(rf_model, X_rf_for_pdp, FEATURE1_NAME, FEATURE2_NAME, TARGET_NAME, OUTPUT_SUBDIR)
            else:
                print("Skipping PDP plot as Random Forest model or its training data is not available.")

    print("\nAdvanced modeling script finished.")

if __name__ == '__main__':
    main()