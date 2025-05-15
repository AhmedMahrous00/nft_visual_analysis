import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from pathlib import Path
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse

SEED = 42
IMG_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)

def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    mean_hsv = np.mean(hsv, axis=(0,1))
    colorfulness = np.std(lab[:,:,1:], axis=(0,1)).mean()
    brightness = np.mean(img)
    
    return {
        'mean_hue': mean_hsv[0],
        'mean_saturation': mean_hsv[1],
        'mean_value': mean_hsv[2],
        'colorfulness': colorfulness,
        'brightness': brightness
    }

def extract_composition_features(img):
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    rule_of_thirds_score = 0
    if corners is not None:
        corners = corners.reshape(-1, 2)
        h_third, w_third = height // 3, width // 3
        rule_of_thirds_score = np.mean(
            (np.abs(corners[:, 0] - w_third) < w_third/4) |
            (np.abs(corners[:, 0] - 2*w_third) < w_third/4) |
            (np.abs(corners[:, 1] - h_third) < h_third/4) |
            (np.abs(corners[:, 1] - 2*h_third) < h_third/4)
        )
    
    h_symmetry = np.mean(np.abs(img - np.flip(img, axis=0)))
    v_symmetry = np.mean(np.abs(img - np.flip(img, axis=1)))
    
    return {
        'rule_of_thirds_score': rule_of_thirds_score,
        'horizontal_symmetry': h_symmetry,
        'vertical_symmetry': v_symmetry
    }

def extract_texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray, 100, 200)
    texture_edge_density = np.mean(edges > 0)
    entropy = shannon_entropy(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = {f'glcm_{prop}': graycoprops(glcm, prop).mean() for prop in props}
    
    return {
        'texture_edge_density': texture_edge_density,
        'entropy': entropy,
        'contrast': contrast,
        'sharpness': sharpness,
        **glcm_features
    }

def extract_structure_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, _ = cv2.connectedComponents(binary)
    num_objects = num_labels - 1
    
    white_space = np.mean(gray > 200)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    spatial_complexity = edge_density * num_objects
    
    return {
        'num_objects': num_objects,
        'white_space_percentage': white_space,
        'spatial_complexity': spatial_complexity
    }

def extract_simplicity_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray, 100, 200)
    simplicity_edge_density = np.mean(edges > 0)
    
    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
    color_simplicity = 1.0 / (1.0 + np.log1p(unique_colors))
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_shapes = len(contours)
    shape_simplicity = 1.0 / (1.0 + np.log1p(num_shapes))
            
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_complexity = np.mean(gradient_magnitude)
    
    return {
        'simplicity_edge_density': simplicity_edge_density,
        'color_simplicity': color_simplicity,
        'shape_simplicity': shape_simplicity,
        'gradient_complexity': gradient_complexity
    }

def extract_lbp_features(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        
        return {
            'lbp_hist_mean': np.mean(hist),
            'lbp_hist_std': np.std(hist)
        }
    except Exception as e:
        print(f"Error in LBP feature extraction: {e}")
        return {'lbp_hist_mean': 0.0, 'lbp_hist_std': 0.0}


def extract_hog_features(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
        
        return {
            'hog_mean': np.mean(fd),
            'hog_std': np.std(fd)
        }
    except Exception as e:
        print(f"Error in HOG feature extraction: {e}")
        return {'hog_mean': 0.0, 'hog_std': 0.0}


def extract_hu_moments_features(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)
        
        features = {}
        for i in range(0, 2):
            features[f'hu_moment_{i+1}'] = hu_moments_log[i,0]
        return features
    except Exception as e:
        print(f"Error in Hu Moments feature extraction: {e}")
        return {f'hu_moment_{i+1}': 0.0 for i in range(2)}

def extract_color_histogram_moments_features(img):
    features = {}
    try:
        color_spaces = {
            'rgb': img
        }

        for space_name, space_img in color_spaces.items():
            for i in range(space_img.shape[2]):
                channel = space_img[:, :, i]
                
                try:
                    channel_skew = skew(channel.ravel())
                    channel_kurtosis = kurtosis(channel.ravel())
                except ValueError as ve:
                    print(f"ValueError during skew/kurtosis for {space_name}_channel_{i}: {ve}")
                    channel_skew = 0.0
                    channel_kurtosis = 0.0

                features[f'{space_name}_channel_{i}_skew'] = channel_skew
                features[f'{space_name}_channel_{i}_kurtosis'] = channel_kurtosis
        return features
    except Exception as e:
        print(f"Error in Color Histogram Moments feature extraction: {e}")
        num_channels = 3 
        color_mom_error_features = {}
        for i in range(num_channels):
            color_mom_error_features[f'rgb_channel_{i}_skew'] = 0.0
            color_mom_error_features[f'rgb_channel_{i}_kurtosis'] = 0.0
        return color_mom_error_features

def extract_saliency_features(img):
    features = {}
    try:
        saliency_algorithm = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency_algorithm.computeSaliency(img)
        
        if success and saliency_map is not None:
            features['saliency_mean'] = np.mean(saliency_map)
            features['saliency_std'] = np.std(saliency_map)
        else:
            features['saliency_mean'] = 0.0
            features['saliency_std'] = 0.0
        return features
    except Exception as e:
        print(f"Error in Saliency feature extraction: {e}")
        return {'saliency_mean': 0.0, 'saliency_std': 0.0}

def extract_all_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    features = {}
    features.update(extract_color_features(img))
    features.update(extract_composition_features(img))
    features.update(extract_texture_features(img))
    features.update(extract_structure_features(img))
    features.update(extract_simplicity_features(img))
    features.update(extract_lbp_features(img))
    features.update(extract_hog_features(img))
    features.update(extract_hu_moments_features(img))
    features.update(extract_color_histogram_moments_features(img))
    features.update(extract_saliency_features(img))
    
    return features

def extract_features_batch(image_paths):
    features_list = []
    for path in tqdm(image_paths, desc="Processing images", leave=False):
        try:
            features = extract_all_features(path)
            features_list.append(features)
        except Exception as e:
            print(f"\nError processing {path}: {e}")
            features_list.append(None)
    return features_list

def process_chunk(chunk_data):
    chunk_paths, chunk_idx = chunk_data
    print(f"\nProcessing chunk {chunk_idx + 1} with {len(chunk_paths)} images...")
    return extract_features_batch(chunk_paths)

def extract_and_save_features(input_metadata, image_base_dir, output_csv_full, output_csv_modeling, max_collections=None, max_per_collection=None):
    print(f"\nLoading and preprocessing data from: {input_metadata}")
    input_path = Path(input_metadata)
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path, low_memory=False)
    else:
        print(f"Error: Unsupported input file format '{input_path.suffix}'. Please use .parquet or .csv")
        exit(1)
    
    if 'collection' in df.columns and 'collection_unique' not in df.columns:
        df = df.rename(columns={'collection': 'collection_unique'})
    
    print("\nHandling NaN values in critical columns...")
    
    text_columns = {
        'name': 'Unnamed NFT',
        'description': '',
        'metadata': '{}'
    }
    for col, fill_value in text_columns.items():
        df[col] = df[col].fillna(fill_value)
    
    numeric_columns = ['rarity', 'rarityRank']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    print(f"Missing last_sale_usd: {df['last_sale_usd'].isnull().sum()}")
    print(f"Zero-valued last_sale_usd: {(df['last_sale_usd']==0).sum()}")
    df = df[df['last_sale_usd'].notnull() & (df['last_sale_usd'] > 0)].copy()
    print(f"Total NFTs after filtering: {len(df)}")
    
    if 'image_path' not in df.columns:
        print("Error: 'image_path' column not found in the input metadata file.")
        exit(1)

    initial_paths = df['image_path'].nunique()
    print(f"\nInitial unique image paths in CSV: {initial_paths}")
    
    if max_collections is not None:
        counts = df['collection_unique'].value_counts()
        top_cols = counts.nlargest(max_collections).index.tolist()
        print(f"Limiting to top {max_collections} collections:", top_cols)
        df = df[df['collection_unique'].isin(top_cols)].reset_index(drop=True)
        print(f"NFTs after limiting collections: {len(df)}")
    
    print("\nChecking image files...")
    tqdm.pandas(desc="Checking image files")
    image_base_dir_path = Path(image_base_dir)
    df['exists'] = df['image_path'].progress_apply(lambda rel: (image_base_dir_path / rel).exists())
    
    found = int(df['exists'].sum())
    missing = int(len(df) - found)
    print(f"Files found:   {found}")
    print(f"Files missing: {missing}")
    
    df = df[df['exists']].copy()
    df['resolved_path'] = df['image_path'].apply(lambda x: str(image_base_dir_path / x))
    df = df.drop('exists', axis=1)
    print(f"NFTs with valid image paths: {len(df)}")
    
    if max_per_collection is not None:
        print(f"\nSampling up to {max_per_collection} NFTs per collection...")
        df = (df.groupby('collection_unique', group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), max_per_collection), random_state=SEED))
              .reset_index(drop=True))
        print(f"Final NFT count after sampling: {len(df)}")
    
    print("\nExtracting features...")
    n_cores = max(1, cpu_count() - 1)
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    chunk_size = max(1, len(df) // (n_cores * 4))
    path_chunks = [(df['resolved_path'].iloc[i:i + chunk_size], i//chunk_size) 
                  for i in range(0, len(df), chunk_size)]
    print(f"Split data into {len(path_chunks)} chunks of size {chunk_size}")
    
    with Pool(n_cores) as pool:
        features_list = []
        for chunk_features in tqdm(pool.imap(process_chunk, path_chunks), 
                                 total=len(path_chunks),
                                 desc="Processing chunks"):
            features_list.extend(chunk_features)
            print(f"Completed chunk with {len(chunk_features)} features")
    
    print("\nConverting features to DataFrame...")
    valid_features = [f for f in features_list if f is not None]
    if len(valid_features) < len(features_list):
        print(f"Warning: {len(features_list) - len(valid_features)} images failed to process")
    features_df = pd.DataFrame(valid_features)
    
    df = df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    print("\nRemoving features with NaN values...")
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        print(f"Removing {len(nan_cols)} features with NaN values: {nan_cols}")
        features_df = features_df.drop(columns=nan_cols)
    
    df = pd.concat([df, features_df], axis=1)
    
    print("\nPreparing price targets...")
    df['log_price'] = np.log1p(df['last_sale_usd'])
    low, high = np.percentile(df['log_price'], [1, 99])
    df['log_price_clipped'] = df['log_price'].clip(lower=low, upper=high)
    
    print(f"\nSaving processed data to: {output_csv_full}")
    Path(output_csv_full).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_full, index=False)
    print(f"Saved final processed dataset (with features) to {output_csv_full}")

    print(f"\nCreating a smaller CSV for modeling, saving to: {output_csv_modeling}")
    base_columns = ['image_path', 'log_price_clipped']
    if 'collection_unique' in df.columns:
        base_columns.insert(1, 'collection_unique')
    visual_features_for_modeling = [
        'mean_hue', 'mean_saturation', 'mean_value', 'colorfulness', 'brightness',
        'rule_of_thirds_score', 'horizontal_symmetry', 'vertical_symmetry',
        'texture_edge_density', 'entropy', 'contrast', 'sharpness',
        'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'num_objects', 'white_space_percentage', 'spatial_complexity',
        'simplicity_edge_density', 'color_simplicity', 'shape_simplicity', 'gradient_complexity',
        'lbp_hist_mean', 'lbp_hist_std', 'hog_mean', 'hog_std',
        'hu_moment_1', 'hu_moment_2',
        'saliency_mean', 'saliency_std'
    ]
    columns_for_modeling_csv = base_columns + visual_features_for_modeling
    existing_columns_for_modeling = [col for col in columns_for_modeling_csv if col in df.columns]
    if not existing_columns_for_modeling:
        print(f"Warning: No columns found for the modeling CSV. Skipping its creation.")
    else:
        modeling_df = df[existing_columns_for_modeling]
        Path(output_csv_modeling).parent.mkdir(parents=True, exist_ok=True)
        modeling_df.to_csv(output_csv_modeling, index=False)
        print(f"Saved selected features for modeling to: {output_csv_modeling}")
        print(f"   Selected {len(modeling_df.columns)} columns: {modeling_df.columns.tolist()}")
    print(f"\nFinal dataset shape (overall): {df.shape}")
    print(f"Number of selected features for modeling CSV: {len(existing_columns_for_modeling) - len(base_columns)}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts handcrafted visual features from NFT images and prepares data for modeling.")
    parser.add_argument('--input_metadata', type=str, default="data/merged_nfts_last_sale.parquet",
                        help="Path to the input Parquet or CSV metadata file. Default: data/merged_nfts_last_sale.parquet")
    parser.add_argument('--image_base_dir', type=str, default="data/images",
                        help="Base directory where image files (referenced in metadata's 'image_path' column) are stored. Default: data/images")
    parser.add_argument('--max_collections', type=int, default=None,
                        help='If set, only process the top N collections by size. Default: None (all collections).')
    parser.add_argument('--max_per_collection', type=int, default=None,
                        help='If set, limit to N NFTs per collection. Default: None (all NFTs per collection).')
    args = parser.parse_args()

    from pathlib import Path
    input_path = Path(args.input_metadata)
    input_stem = input_path.stem
    output_csv_full = f"outputs/traditional/{input_stem}_processed_with_all_features.csv"
    output_csv_modeling = f"outputs/traditional/summaries/{input_stem}_selected_features_for_modeling.csv"

    extract_and_save_features(
        args.input_metadata,
        args.image_base_dir,
        output_csv_full,
        output_csv_modeling,
        args.max_collections,
        args.max_per_collection
    )
    print("\nFeature extraction script finished.")