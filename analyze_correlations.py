import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr, spearmanr
import os

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between features and Fisher's z.")
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True,
        help='Path to the input CSV file containing Fisher\'s z and aggregate features.'
    )

    args = parser.parse_args()

    output_csv = "outputs/experiment_correlations/feature_fisher_z_correlations.csv"
    target_col_original = "log_price_clipped"

    print(f"Loading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if 'fisher_z' not in df.columns:
        print("Error: 'fisher_z' column not found in the input CSV.")
        return

    # Identify the 192 aggregate visual feature columns
    excluded_cols = ['collection_unique', 'rc', 'fisher_z']
    visual_feature_columns = [
        col for col in df.columns 
        if col not in excluded_cols and not col.startswith(target_col_original)
    ]

    price_aggregate_columns = [
        f'{target_col_original}_mean',
        f'{target_col_original}_median',
        f'{target_col_original}_var'
    ]
    price_aggregate_columns = [col for col in price_aggregate_columns if col in df.columns]

    all_features_to_correlate = visual_feature_columns + price_aggregate_columns
    
    print(f"Identified {len(visual_feature_columns)} visual features and {len(price_aggregate_columns)} price aggregate features.")
    print(f"Total features to correlate with 'fisher_z': {len(all_features_to_correlate)}.")
    
    if not all_features_to_correlate:
        print("No features identified for correlation. Exiting.")
        return

    correlations_list = []
    target_fisher_z = df['fisher_z']

    for feature in all_features_to_correlate:
        current_feature_data = df[feature]
        
        valid_indices = target_fisher_z.notna() & current_feature_data.notna()
        clean_fisher_z = target_fisher_z[valid_indices]
        clean_feature_data = current_feature_data[valid_indices]

        pearson_corr_val = np.nan
        pearson_p_value = np.nan
        spearman_corr_val = np.nan
        spearman_p_value = np.nan

        if len(clean_fisher_z) >= 2:
            try:
                pearson_corr_val, pearson_p_value = pearsonr(clean_feature_data, clean_fisher_z)
            except Exception as e:
                print(f"Could not calculate Pearson correlation for {feature}: {e}")
            try:
                spearman_corr_val, spearman_p_value = spearmanr(clean_feature_data, clean_fisher_z)
            except Exception as e:
                print(f"Could not calculate Spearman correlation for {feature}: {e}")
        
        correlations_list.append({
            'feature': feature,
            'pearson_correlation': pearson_corr_val,
            'pearson_p_value': pearson_p_value,
            'spearman_correlation': spearman_corr_val,
            'spearman_p_value': spearman_p_value
        })

    correlations_df = pd.DataFrame(correlations_list)
    correlations_df['abs_pearson_correlation'] = correlations_df['pearson_correlation'].abs()
    correlations_df['abs_spearman_correlation'] = correlations_df['spearman_correlation'].abs()
    correlations_df.sort_values(by='abs_spearman_correlation', ascending=False, inplace=True)
    
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    try:
        correlations_df.to_csv(output_csv, index=False)
        print(f"Feature correlations with Fisher's z saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving correlations CSV: {e}")

    print("\nTop 20 features by absolute Spearman correlation with Fisher's z:")
    print(correlations_df[['feature', 'pearson_correlation', 'pearson_p_value', 'spearman_correlation', 'spearman_p_value']].head(20).to_string())

if __name__ == '__main__':
    main() 