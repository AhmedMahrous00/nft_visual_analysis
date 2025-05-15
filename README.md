# Replication Repo: Controlling Collection Leakage to Reveal Genuine Aesthetic Effects in NFT Pricing

This repository contains the code to replicate the results presented in the paper "Controlling Collection Leakage to Reveal Genuine Aesthetic Effects in NFT Pricing".

## 1. Setup

### Prerequisites
- Python (version 3.8+ recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedMahrous00/nft_visual_analysis.git
   cd nft_visual_analysis
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Data Preparation

1.  **Place the main NFT metadata file** (e.g., `merged_nfts_last_sale.parquet`) into a directory (e.g., `data/`).
2.  **Place all corresponding NFT image files** into a subdirectory within `data/images/`. Ensure the paths in your metadata file correctly reference these images (e.g., if `image_path` in the parquet is `collection_A/image1.png`, then the image should be at `data/images/collection_A/image1.png`).

## 3. Replication Steps

The following scripts should be run in order from the root of the project directory.

### Step 1: Extract Handcrafted Features
This script processes the initial dataset, extracts handcrafted visual features, and prepares a CSV for subsequent analysis.

```bash
python nft_analysis/extract_handcrafted_features.py
```
-   **Input:** `data/merged_nfts_last_sale.parquet` and images in `data/images/`
-   **Main Output:** `outputs/traditional/summaries/selected_features_for_modeling.csv` (This CSV will be used by many of the following scripts).

### Step 2: Train Handcrafted Machine Learning Models
This script trains and evaluates traditional ML models (e.g., Ridge, LightGBM) using the handcrafted features.

```bash
python nft_analysis/train_traditional_models.py --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv
```
-   **Input:** `outputs/traditional/summaries/selected_features_for_modeling.csv`
-   **Main Output:** Evaluation results (JSON files) in `outputs/traditional_simple/summaries/`

### Step 3: Extract Deep Learning Features and Train Models
This script extracts features using various deep learning models and then trains LightGBM models on these deep features.

```bash
python nft_analysis/extract_deep_features.py --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv
```
-   **Input:** `outputs/traditional/summaries/selected_features_for_modeling.csv` and images in `data/images/`
-   **Main Outputs:**
    -   Deep features (`.npy` files) in `outputs/feature_lgbm_pipeline/features/`
    -   Evaluation results (CSVs/JSONs) in `outputs/feature_lgbm_pipeline/summaries/`

### Step 4: Analyze Collection-Level Photorelevance and Aggregate Features
This script calculates the predictive performance (correlation `rc` and Fisher's Z-transformed `fisher_z`) of a model within each NFT collection using handcrafted features. It then computes aggregate statistics (mean, median, variance) for each visual feature and the price within each collection, log-transforms these aggregates, and performs PCA. It also calculates VIF and correlation matrices for these aggregate features.

```bash
python nft_analysis/correlation_analysis.py --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv
```
-   **Input:** `outputs/traditional/summaries/selected_features_for_modeling.csv` (from Step 1)
-   **Main Outputs:**
    -   `outputs/experiment_correlations/collection_correlations.csv`: Per-collection `rc` and `fisher_z` scores.
    -   `outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv`: Key file containing `fisher_z` merged with log-transformed aggregate features for each collection. This is used as input for the next step.
    -   `outputs/experiment_correlations/aggregate_features_correlation_matrix.csv`: Correlation matrix of the aggregate features.
    -   `outputs/experiment_correlations/aggregate_features_vif_scores.csv`: VIF scores for aggregate features.

### Step 5: Initial Correlation of Aggregate Features with Fisher's Z

This script takes the aggregate features and Fisher's Z scores for each collection (from `collection_regression_data_with_fisher_z.csv`) and calculates the Pearson and Spearman correlations between each aggregate feature and `fisher_z`.

```bash
# Example if it were a separate script:
# python nft_analysis/analyze_aggregate_feature_correlations.py --input_csv outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv --output_csv outputs/experiment_correlations/feature_fisher_z_correlations.csv
```
-   **Input:** `outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv`
-   **Main Output:** `outputs/experiment_correlations/feature_fisher_z_correlations.csv`: Contains columns listing correlations of aggregate features with `fisher_z` and the statistical significance of these correlations.

### Step 6: Photorelevance Feature Selection
This script uses the results from the previous steps to perform feature selection and build models to predict the collection-level photorelevance (`fisher_z`) using aggregate collection features.

```bash
python nft_analysis/correlation_features.py --correlations_input_csv outputs/experiment_correlations/feature_fisher_z_correlations.csv --main_data_input_csv outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv
```
-   **Inputs:**
    -   `outputs/experiment_correlations/feature_fisher_z_correlations.csv` (from Step 5; lists initial correlations of aggregate features with `fisher_z`)
    -   `outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv` (from Step 4; contains `fisher_z` and the aggregate features per collection)
-   **Main Outputs (in `outputs/advanced_modeling/`):**
    -   `candidate_features_correlations.csv`: Filtered candidate aggregate features.
    -   `bootstrap_feature_selection_counts.csv`: Counts of how often each aggregate feature was selected by bootstrap Elastic Net.
    -   `topN_features_correlation_matrix.csv`: Correlation matrix of the top N selected aggregate features.
    -   `ols_summary_topN.txt` & `ols_pred_vs_actual_topN.png`: OLS regression results on top N aggregate features.
    -   `rf_permutation_importance_topN.csv` & `rf_r2_scores_topN.txt`: Random Forest results and feature importance for top N aggregate features.
    
## 4. Expected Outputs
After running all steps, the `outputs/` directory will contain:
-   Extracted features (handcrafted and deep)
-   Model training results
-   Correlation analyses and other experimental results


## 5. Citation
If you use this code or data in your research, please cite our paper:
```
[Your Full Paper Citation Here - e.g., Author, A., Author, B. (Year). Paper Title. Journal/Conference.]
```

## Contact
For inquiries or issues, please contact Ahmed Mahrous at ahmed.mahrous@kaust.edu.sa.
