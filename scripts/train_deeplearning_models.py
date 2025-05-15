import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import timm
import clip

SEED = 42
K_FOLDS_COMPARISON = 10
K_FOLDS_FINAL = 10
IMG_SIZE = 224
BATCH_SIZE_FEATURES = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed(SEED)

def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

class NFTImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path)
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA').convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            if self.transform:
                img = self.transform(img)
            return img, img_path
        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            placeholder_img = torch.zeros((3, IMG_SIZE, IMG_SIZE)) 
            return placeholder_img, img_path

def load_and_preprocess_data(csv_path, base_image_dir=None, max_collections=None, max_per_collection=None, save_pickle=True, output_dir_path=None):
    print("Loading and preprocessing data...")
    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = ['collection_unique', 'log_price_clipped']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in {csv_path}.")
            exit(1)

    image_col = None
    if 'image_file' in df.columns:
        image_col = 'image_file'
        df['full_image_path'] = df[image_col].astype(str).str.replace('\\', '/', regex=False)
    elif 'image_path' in df.columns:
         image_col = 'image_path'
         df['full_image_path'] = df[image_col].astype(str).str.replace('\\', '/', regex=False)
    else:
        print("Error: Neither 'image_file' nor 'image_path' column found in CSV.")
        exit(1)

    df = df[df['log_price_clipped'].notnull()].copy()

    tqdm.pandas(desc="Validating image file paths")
    mask_file_exists = df['full_image_path'].progress_apply(lambda p: Path(p).is_file() if p else False)

    dropped_count = len(df) - mask_file_exists.sum()
    if dropped_count > 0:
        print(f"Dropping {dropped_count} rows whose full_image_path is not a valid file.")
        if not mask_file_exists.all():
             print("Examples of problematic paths:", df[~mask_file_exists]['full_image_path'].head().tolist())
    df = df[mask_file_exists].reset_index(drop=True)

    if df.empty:
        print("DataFrame is empty after preprocessing and filtering. Check paths and filters.")
        return pd.DataFrame()

    if max_collections is not None and not df.empty:
        print(f"Limiting to top {max_collections} collections based on 'collection_unique' counts.")
        top_n_collections = df['collection_unique'].value_counts().nlargest(max_collections).index
        df = df[df['collection_unique'].isin(top_n_collections)].reset_index(drop=True)

    if max_per_collection is not None and not df.empty:
        print(f"Sampling max {max_per_collection} NFTs per 'collection_unique'.")
        df = (df.groupby('collection_unique', group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_per_collection), random_state=SEED))
                .reset_index(drop=True))

    if df.empty:
        print("DataFrame is empty after preprocessing and filtering. Check paths and filters.")
        return pd.DataFrame()

    print(f"Final collections retained (based on collection_unique): {df['collection_unique'].nunique()}")
    print(f"Final NFT count for feature extraction: {len(df)}")

    df['unique_id'] = df['full_image_path']

    if save_pickle and output_dir_path:
        pickle_path = output_dir_path / 'df_preprocessed_for_features.pkl'
        df.to_pickle(pickle_path)
        print(f"Saved preprocessed DataFrame to {pickle_path}")
    return df

MODELS_CONFIG = [
    {
        'name': 'convnext_base',
        'type': 'timm',
        'timm_name': 'convnext_base.fb_in22k_ft_in1k',
        'input_size_override': None
    },
    {
        'name': 'resnet50',
        'type': 'timm',
        'timm_name': 'resnet50',
        'input_size_override': None
    },
    {
        'name': 'beit_base_patch16_224',
        'type': 'timm',
        'timm_name': 'beit_base_patch16_224.in22k_ft_in22k_in1k',
        'input_size_override': 224
    },
    {
        'name': 'clip_vitb16',
        'type': 'clip',
        'clip_name': 'ViT-B/16',
    },
    {
        'name': 'swin_tiny_patch4_window7_224',
        'type': 'timm',
        'timm_name': 'swin_tiny_patch4_window7_224.ms_in1k',
        'input_size_override': 224
    }
]

def get_model_and_transform(config, device):
    model = None
    transform = None
    model_name = config['name']
    print(f"Loading model and transform for: {model_name}...")

    if config['type'] == 'clip':
        if not clip:
            raise ImportError("OpenAI CLIP library is not installed. Please install it.")
        model, transform = clip.load(config['clip_name'], device=device, jit=False)

    elif config['type'] == 'timm':
        if not timm:
            raise ImportError("timm library is not installed. Please install it.")

        timm_model_name = config['timm_name']
        print(f"  Using timm model: {timm_model_name}")
        model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        model = model.to(device)
        model.eval()

        data_config = timm.data.resolve_model_data_config(model)
        if config.get('input_size_override'):
            data_config['input_size'] = (3, config['input_size_override'], config['input_size_override'])

        print(f"  Resolved data config for {model_name}: {data_config}")
        transform = timm.data.create_transform(**data_config, is_training=False)
        print(f"  Transform for {model_name}: {transform}")

    else:
        raise ValueError(f"Unknown model type in config: {config.get('type')} for model {model_name}")

    if model is None or transform is None:
        raise RuntimeError(f"Failed to load model or transform for {model_name}")

    return model, transform


def extract_and_save_features(model_config_entry, df_to_process, current_batch_size, current_device, current_features_dir):
    model_name = model_config_entry['name']
    features_path = current_features_dir / f"{model_name}_features.npy"
    ids_path = current_features_dir / f"{model_name}_ids.npy"

    print(f"\n--- Starting feature extraction for: {model_name} ---")

    if features_path.exists() and ids_path.exists() and not args.force_feature_extraction:
        print(f"Features for {model_name} already exist. Skipping extraction.")
        return str(features_path), str(ids_path)

    model, transform = get_model_and_transform(model_config_entry, current_device)

    if 'unique_id' not in df_to_process.columns or 'full_image_path' not in df_to_process.columns:
        raise ValueError("DataFrame must contain 'unique_id' and 'full_image_path' columns.")

    dataset = NFTImageDataset(
        image_paths=df_to_process['full_image_path'].tolist(),
        transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    all_features = []
    all_original_paths_processed = []

    with torch.no_grad():
        for batch_idx, (imgs, batch_img_paths) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Extracting {model_name}"):
            imgs = imgs.to(current_device)

            valid_indices = [i for i, img_path in enumerate(batch_img_paths) if not (imgs[i].ndim == 3 and torch.all(imgs[i] == 0))]

            if not valid_indices:
                continue

            imgs_valid = imgs[valid_indices]
            batch_img_paths_valid = [batch_img_paths[i] for i in valid_indices]

            if imgs_valid.numel() == 0:
                 continue


            if model_config_entry['type'] == 'clip':
                features = model.encode_image(imgs_valid)
            else:
                features = model(imgs_valid)

            all_features.append(features.cpu().numpy())
            all_original_paths_processed.extend(batch_img_paths_valid)

    if not all_features:
        print(f"Error: No features were extracted for {model_name}. Check image paths and model compatibility.")
        return None, None

    concatenated_features = np.vstack(all_features)

    path_to_id_map = pd.Series(df_to_process.unique_id.values, index=df_to_process.full_image_path).to_dict()

    final_ids_for_saved_features = [path_to_id_map.get(p) for p in all_original_paths_processed]
    final_ids_for_saved_features = [fid for fid in final_ids_for_saved_features if fid is not None]

    if len(final_ids_for_saved_features) != concatenated_features.shape[0]:
        print(f"Warning: Mismatch between number of features ({concatenated_features.shape[0]}) "
              f"and number of mapped IDs ({len(final_ids_for_saved_features)}) for {model_name}. "
              f"This should not happen if all_original_paths_processed are derived from df_to_process['full_image_path'].")

    np.save(features_path, concatenated_features)
    np.save(ids_path, np.array(final_ids_for_saved_features))

    print(f"Saved {model_name} features ({concatenated_features.shape}) to {features_path}")
    print(f"Saved {model_name} IDs ({len(final_ids_for_saved_features)}) to {ids_path}")

    del model
    del transform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.lgbm_n_jobs != -1:
        LGBM_PARAMS['n_jobs'] = args.lgbm_n_jobs
        print(f"Set LightGBM n_jobs to: {args.lgbm_n_jobs}")

    return str(features_path), str(ids_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NFT Feature Extraction and LightGBM Price Prediction Pipeline")
    parser.add_argument('--input_csv', type=str,
                        default="outputs/traditional/summaries/selected_features_for_modeling.csv",
                        help='Path to the input CSV file (should contain collection_unique column)')
    parser.add_argument('--image_base_dir', type=str,
                        default="data/images",
                        help='Base directory where NFT images are stored (for reference, not used in this script)')

    args = parser.parse_args()

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.lgbm_n_jobs != -1:
        LGBM_PARAMS['n_jobs'] = args.lgbm_n_jobs
        print(f"Set LightGBM n_jobs to: {args.lgbm_n_jobs}")

    if not timm or not clip:
        print("Error: 'timm' or 'openai-clip' library is not installed. This script requires them. Please install and try again.")
        exit()

    OUTPUT_DIR = Path(args.output_dir)
    FEATURES_SUBDIR = OUTPUT_DIR / "features"
    SUMMARIES_SUBDIR = OUTPUT_DIR / "summaries"
    MODELS_SUBDIR = OUTPUT_DIR / "models_lgbm"
    FEATURES_SUBDIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_SUBDIR.mkdir(parents=True, exist_ok=True)
    MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)

    df_full = load_and_preprocess_data(
        csv_path=args.input_csv,
        base_image_dir=args.image_base_dir,
        max_collections=max_collections,
        max_per_collection=max_per_collection,
        save_pickle=True,
        output_dir_path=OUTPUT_DIR
    )

    if args.sample_fraction is not None and 0 < args.sample_fraction <= 1:
        initial_rows = len(df_full)
        df_full = df_full.sample(frac=args.sample_fraction, random_state=SEED).reset_index(drop=True)
        print(f"Applied sampling: Reduced data from {initial_rows} rows to {len(df_full)} rows ({args.sample_fraction*100:.2f}%)")
    elif args.sample_fraction is not None and (args.sample_fraction <= 0 or args.sample_fraction > 1):
        print(f"Warning: Invalid sample_fraction value: {args.sample_fraction}. Must be between 0 and 1. No sampling applied.")
        args.sample_fraction = None

    if df_full.empty:
        print("No data to process after loading and preprocessing. Exiting.")
        exit()

    print(f"Data loaded: {len(df_full)} items with resolved image paths for feature extraction.")

    print("\n--- Stage 2: Feature Extraction ---")
    extracted_feature_files = {}

    if df_full.empty or 'full_image_path' not in df_full.columns or df_full['full_image_path'].nunique() == 0:
        print("No valid image paths in DataFrame to extract features from. Skipping feature extraction.")
    else:
        if 'unique_id' not in df_full.columns:
            df_full['unique_id'] = df_full['full_image_path']
            print("Created 'unique_id' column from 'full_image_path' for feature mapping.")

        for model_conf in MODELS_CONFIG:
            model_name = model_conf['name']

            feat_path = FEATURES_SUBDIR / f"{model_name}_features.npy"
            id_path = FEATURES_SUBDIR / f"{model_name}_ids.npy"

            if not args.force_feature_extraction and feat_path.exists() and id_path.exists():
                print(f"Features for {model_name} found at {feat_path}. Skipping extraction.")
                extracted_feature_files[model_name] = {'features': str(feat_path), 'ids': str(id_path)}
            elif args.skip_feature_extraction :
                 print(f"Skipping feature extraction for {model_name} due to --skip_feature_extraction flag.")
                 if feat_path.exists() and id_path.exists():
                    extracted_feature_files[model_name] = {'features': str(feat_path), 'ids': str(id_path)}
                 else:
                    print(f"Warning: --skip_feature_extraction is set, but feature files for {model_name} do not exist at {feat_path}.")
            else:
                print(f"Extracting features for {model_name}...")
                f_path, i_path = extract_and_save_features(
                    model_config_entry=model_conf,
                    df_to_process=df_full[['unique_id', 'full_image_path']].drop_duplicates().reset_index(drop=True),
                    current_batch_size=BATCH_SIZE_FEATURES,
                    current_device=DEVICE,
                    current_features_dir=FEATURES_SUBDIR
                )
                if f_path and i_path:
                    extracted_feature_files[model_name] = {'features': f_path, 'ids': i_path}
                else:
                    print(f"Failed to extract features for {model_name}.")

    if not extracted_feature_files:
        print("\nNo features were extracted or found. Cannot proceed to LightGBM training. Exiting.")
        exit()

    print(f"\nSummary of feature files available for LightGBM training:")
    for model_name, paths in extracted_feature_files.items():
        print(f"  {model_name}: Features at {paths['features']}, IDs at {paths['ids']}")


    print("\nFeature extraction setup and main loop implemented.")

    LGBM_PARAMS = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'early_stopping_round': 50
    }

    def load_features_and_merge(df_metadata, model_name, feature_file_paths, id_col_name='unique_id'):
        print(f"\nLoading features for {model_name}...")
        features_path = feature_file_paths.get('features')
        ids_path = feature_file_paths.get('ids')

        if not features_path or not ids_path or not Path(features_path).exists() or not Path(ids_path).exists():
            print(f"Error: Feature or ID file missing for {model_name}. Paths: {features_path}, {ids_path}")
            return None, []

        try:
            features_array = np.load(features_path)
            ids_array = np.load(ids_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading .npy files for {model_name}: {e}")
            return None, []

        if features_array.shape[0] != ids_array.shape[0]:
            print(f"Error: Mismatch in number of samples between features ({features_array.shape[0]}) "
                  f"and IDs ({ids_array.shape[0]}) for {model_name}.")
            return None, []

        feature_cols = [f'{model_name}_feat_{i}' for i in range(features_array.shape[1])]
        df_features = pd.DataFrame(features_array, columns=feature_cols)
        df_features[id_col_name] = ids_array

        scaler = StandardScaler()
        df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])
        print(f"  Standardized {len(feature_cols)} features for {model_name}.")

        df_merged = pd.merge(df_metadata, df_features, on=id_col_name, how='inner')

        if df_merged.empty:
            print(f"Warning: DataFrame is empty after merging features for {model_name}. "
                  f"This could be due to ID mismatches or no common IDs. Original df_metadata: {len(df_metadata)}, features loaded: {len(df_features)}.")
            return None, []

        print(f"  Successfully loaded and merged features for {model_name}. Shape of merged df: {df_merged.shape}")
        return df_merged, feature_cols

    def run_lgbm_cv(df_train_val, feature_columns, target_column, cv_strategy, num_folds,
                    lgbm_model_params, collection_col='collection_unique',
                    model_save_dir=None, model_file_prefix=None, collection_name=None):

        fold_r2_scores = []
        oof_preds = np.zeros(len(df_train_val))

        X = df_train_val[feature_columns]
        y = df_train_val[target_column]

        if cv_strategy == 'pooled':
            print(f"  Running Pooled CV ({num_folds} folds)")
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
            folds_iterator = kf.split(X, y)
            groups = None
        elif cv_strategy == 'cross':
            print(f"  Running Cross-Collection CV ({num_folds} folds)")
            if collection_col not in df_train_val.columns:
                raise ValueError(f"Collection column '{collection_col}' not found for Cross-Collection CV.")
            price_bin_col = 'price_bin'
            if price_bin_col not in df_train_val.columns:
                print(f"Warning: '{price_bin_col}' not found. Using standard GroupKFold for cross-collection.")
                num_bins = min(num_folds, len(y.unique())) if len(y.unique()) >= num_folds else 2
                if num_bins > 1:
                     df_train_val[price_bin_col] = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
                else:
                     df_train_val[price_bin_col] = 0

            group_kf = GroupKFold(n_splits=num_folds)
            groups = df_train_val[collection_col]
            folds_iterator = group_kf.split(X, y, groups)
        elif cv_strategy == 'within':
            raise NotImplementedError("'within' CV strategy should be handled by a wrapper that calls this function per collection for pooled logic.")
        else:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

        for fold_num, (train_idx, val_idx) in tqdm(enumerate(folds_iterator), total=num_folds):
            print(f"    Fold {fold_num + 1}/{num_folds}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            lgbm = lgb.LGBMRegressor(**lgbm_model_params)
            lgbm.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       eval_metric='rmse',
                       callbacks=[lgb.early_stopping(stopping_rounds=lgbm_model_params.get('early_stopping_round', 50), verbose=-1)]
                      )

            preds_val = lgbm.predict(X_val)
            oof_preds[val_idx] = preds_val
            fold_r2 = r2_score(y_val, preds_val)
            fold_r2_scores.append(fold_r2)
            print(f"      Fold {fold_num + 1} R2: {fold_r2:.4f}")

            if model_save_dir and model_file_prefix:
                final_model_save_dir = model_save_dir
                current_model_file_prefix = model_file_prefix

                if cv_strategy == 'within' and collection_name:
                    safe_collection_name = sanitize_filename(collection_name)
                    final_model_save_dir = model_save_dir / safe_collection_name
                if not final_model_save_dir.exists():
                    os.makedirs(final_model_save_dir, exist_ok=True)

                model_path = final_model_save_dir / f"{current_model_file_prefix}_fold{fold_num + 1}.txt"
                lgbm.booster_.save_model(str(model_path))
                print(f"      Saved model to {model_path}")

        mean_r2 = np.mean(fold_r2_scores)
        print(f"    Mean R2 over {num_folds} folds: {mean_r2:.4f}")

        results = {
            'fold_r2_scores': fold_r2_scores,
            'mean_r2': mean_r2,
            'oof_r2': r2_score(y, oof_preds)
        }
        return results

    print("\n--- Stage 3: Initial Comparison of Feature Sets (Pooled CV) ---")
    comparison_results = {}
    best_feature_model_name = None
    best_r2_score = -np.inf

    if not extracted_feature_files:
        print("No feature files found to run comparison. Exiting Stage 3.")
    else:
        for model_name, file_paths in extracted_feature_files.items():
            print(f"\nEvaluating LightGBM with features from: {model_name}")
            df_merged, feature_cols = load_features_and_merge(
                df_metadata=df_full.copy(),
                model_name=model_name,
                feature_file_paths=file_paths,
                id_col_name='unique_id'
            )

            if df_merged is not None and not df_merged.empty and feature_cols:

                pooled_cv_results = run_lgbm_cv(
                    df_train_val=df_merged,
                    feature_columns=feature_cols,
                    target_column='log_price_clipped',
                    cv_strategy='pooled',
                    num_folds=K_FOLDS_COMPARISON,
                    lgbm_model_params=LGBM_PARAMS,
                    model_save_dir=MODELS_SUBDIR / model_name,
                    model_file_prefix=f"{model_name}_pooled_comparison"
                )
                comparison_results[model_name] = pooled_cv_results
                print(f"  {model_name} (Pooled CV {K_FOLDS_COMPARISON}-fold) Mean R2: {pooled_cv_results['mean_r2']:.4f}")

                if pooled_cv_results['mean_r2'] > best_r2_score:
                    best_r2_score = pooled_cv_results['mean_r2']
                    best_feature_model_name = model_name
            else:
                print(f"Skipping LightGBM for {model_name} due to issues in loading/merging features.")

        if comparison_results:
            comparison_summary_filename = f"feature_sets_pooled_comparison_{TIMESTAMP}.json"
            comparison_summary_path = SUMMARIES_SUBDIR / comparison_summary_filename
            with open(comparison_summary_path, 'w') as f:
                json.dump(comparison_results, f, indent=4)
            print(f"\nSaved pooled CV comparison results to {comparison_summary_path}")

            if best_feature_model_name:
                print(f"\nBest feature set from initial comparison: {best_feature_model_name} (Mean R2: {best_r2_score:.4f})")
            else:
                print("\nCould not determine a best feature set from the comparison.")

    print(f"\n--- Stage 4: Full Evaluation of Best Feature Set ({best_feature_model_name}) ---")
    final_evaluation_results = {}

    if best_feature_model_name and best_feature_model_name in extracted_feature_files:
        df_best_features_merged, best_feature_cols = load_features_and_merge(
            df_metadata=df_full.copy(),
            model_name=best_feature_model_name,
            feature_file_paths=extracted_feature_files[best_feature_model_name],
            id_col_name='unique_id'
        )

        if df_best_features_merged is not None and not df_best_features_merged.empty and best_feature_cols:

            print(f"\nRunning Pooled CV (Final - {K_FOLDS_FINAL} folds) for {best_feature_model_name}...")
            pooled_final_results = run_lgbm_cv(
                df_train_val=df_best_features_merged.copy(),
                feature_columns=best_feature_cols,
                target_column='log_price_clipped',
                cv_strategy='pooled',
                num_folds=K_FOLDS_FINAL,
                lgbm_model_params=LGBM_PARAMS,
                model_save_dir=MODELS_SUBDIR / best_feature_model_name,
                model_file_prefix=f"{best_feature_model_name}_pooled_final"
            )
            final_evaluation_results['pooled'] = pooled_final_results
            print(f"  {best_feature_model_name} (Pooled CV {K_FOLDS_FINAL}-fold) Mean R2: {pooled_final_results['mean_r2']:.4f}")

            print(f"\nRunning Cross-Collection CV (Final - {K_FOLDS_FINAL} folds) for {best_feature_model_name}...")
            if 'collection_unique' not in df_best_features_merged.columns:
                print(f"Error: 'collection_unique' column not found in the merged DataFrame for Cross-Collection CV. Skipping.")
                final_evaluation_results['cross_collection'] = {'error': "'collection_unique' column missing in merged data.''''''''''''''"}
            else:
                cross_final_results = run_lgbm_cv(
                    df_train_val=df_best_features_merged.copy(),
                    feature_columns=best_feature_cols,
                    target_column='log_price_clipped',
                    cv_strategy='cross',
                    num_folds=K_FOLDS_FINAL,
                    lgbm_model_params=LGBM_PARAMS,
                    collection_col='collection_unique',
                    model_save_dir=MODELS_SUBDIR / best_feature_model_name,
                    model_file_prefix=f"{best_feature_model_name}_cross_final"
                )
                final_evaluation_results['cross_collection'] = cross_final_results
                print(f"  {best_feature_model_name} (Cross-Collection CV {K_FOLDS_FINAL}-fold, based on 'collection_unique') Mean R2: {cross_final_results.get('mean_r2', float('nan')):.4f}")

            print(f"\nRunning Within-Collection CV (Final - {K_FOLDS_FINAL} folds) for {best_feature_model_name}...")
            within_collection_overall_results = {'collections': {}, 'mean_of_means_r2': None, 'weighted_mean_r2': None}
            collection_mean_r2s = []
            collection_lengths = []

            if 'collection_unique' not in df_best_features_merged.columns:
                print(f"Error: 'collection_unique' column not found in the merged DataFrame for Within-Collection CV. Skipping.")
                final_evaluation_results['within_collection'] = {'error': "'collection_unique' column missing in merged data.''''''''''''''"}
            else:
                grouped_by_collection = df_best_features_merged.groupby('collection_unique')
                for unique_name, group_df in tqdm(grouped_by_collection, desc="Processing Collections (Within CV based on unique_name)"):

                    reported_name_for_metrics = unique_name

                    print(f"  Processing Collection: {unique_name} ({len(group_df)} items)")

                    if len(group_df) < K_FOLDS_FINAL:
                        print(f"    Skipping {unique_name}, not enough samples ({len(group_df)}) for {K_FOLDS_FINAL}-fold CV.")
                        within_collection_overall_results['collections'][unique_name] = {
                            'error': 'insufficient_samples',
                            'num_samples': len(group_df),
                            'collection_name_reported': unique_name
                        }
                        continue

                    current_coll_lgbm_params = LGBM_PARAMS.copy()

                    collection_cv_results = run_lgbm_cv(
                        df_train_val=group_df.copy(),
                        feature_columns=best_feature_cols,
                        target_column='log_price_clipped',
                        cv_strategy='pooled',
                        num_folds=K_FOLDS_FINAL,
                        lgbm_model_params=current_coll_lgbm_params,
                        model_save_dir=MODELS_SUBDIR / best_feature_model_name,
                        model_file_prefix=f"{best_feature_model_name}_within",
                        collection_name=unique_name
                    )

                    if isinstance(collection_cv_results, dict):
                        collection_cv_results_to_store = collection_cv_results.copy()
                        collection_cv_results_to_store['collection_name_reported'] = unique_name
                        within_collection_overall_results['collections'][unique_name] = collection_cv_results_to_store
                    else:
                        print(f"Warning: Unexpected result type for collection {unique_name}. Storing as is.")
                        within_collection_overall_results['collections'][unique_name] = collection_cv_results

                    if isinstance(collection_cv_results, dict) and 'mean_r2' in collection_cv_results:
                        collection_mean_r2s.append(collection_cv_results['mean_r2'])
                        collection_lengths.append(len(group_df))
                        print(f"    {unique_name} (Within-Collection CV {K_FOLDS_FINAL}-fold) Mean R2: {collection_cv_results['mean_r2']:.4f}")
                    elif isinstance(collection_cv_results, dict):
                        print(f"    {unique_name} (Within-Collection CV {K_FOLDS_FINAL}-fold) Mean R2: N/A (key missing)")
                    else:
                        print(f"    {unique_name} (Within-Collection CV {K_FOLDS_FINAL}-fold) Mean R2: N/A (unexpected type)")

                if collection_mean_r2s:
                    within_collection_overall_results['mean_of_means_r2'] = np.mean(collection_mean_r2s)
                    within_collection_overall_results['weighted_mean_r2'] = np.average(collection_mean_r2s, weights=collection_lengths)
                    print(f"  {best_feature_model_name} (Within-Collection Overall) Mean of Collection Means R2: {within_collection_overall_results['mean_of_means_r2']:.4f}")
                    print(f"  {best_feature_model_name} (Within-Collection Overall) Weighted Mean R2: {within_collection_overall_results['weighted_mean_r2']:.4f}")

                final_evaluation_results['within_collection'] = within_collection_overall_results

                final_summary_filename = f"final_evaluation_{best_feature_model_name}_{TIMESTAMP}.json"
                final_summary_path = SUMMARIES_SUBDIR / final_summary_filename
                with open(final_summary_path, 'w') as f:
                    json.dump(final_evaluation_results, f, indent=4)
                print(f"\nSaved final evaluation results for {best_feature_model_name} to {final_summary_path}")
        else:
            print(f"Could not load/merge features for the best model ({best_feature_model_name}). Skipping final evaluation.")
    else:
        print("No best feature model identified from comparison, or no features available. Skipping final detailed evaluation.")

    print("\nLightGBM training and evaluation stages implemented.")
    print("\nFull pipeline script finished.")

def sanitize_filename(name):
    keepchars = (' ', '.', '_', '-')
    return "".join(c for c in str(name) if c.isalnum() or c in keepchars).rstrip()