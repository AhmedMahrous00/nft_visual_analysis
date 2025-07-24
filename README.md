![BCCA 2025 Accepted](https://img.shields.io/badge/Conference-BCCA%202025-success)  
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16415242.svg)](https://doi.org/10.5281/zenodo.16415242)  
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16414740.svg)](https://doi.org/10.5281/zenodo.16414740)

# Demystifying the Role of Aesthetics in NFT Pricing: Model, Analysis, and Insights  
## Code & Data

This repository contains all code and scripts to reproduce the experiments in the paper *Demystifying the Role of Aesthetics in NFT Pricing* (BCCA 2025), plus links to the 35 GB NFT image dataset and metadata.

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/AhmedMahrous00/nft_visual_analysis.git
cd nft_visual_analysis
pip install -r requirements.txt
```

### 2. Download & Unzip Data

```bash
# 35 GB dataset (images + metadata) on Zenodo
wget -O data/nft_data.zip "https://zenodo.org/record/16414740/files/nft_images.zip?download=1"

# Unpack everything into data/
unzip data/nft_data.zip -d data/

# After unzipping you should see:
# data/nft_metadata.parquet
# data/images/  ← 32 subfolders of PNGs
```

### 3. Replication Steps

Run the scripts in order from the project root:

```bash
# 0) Convert images (if you downloaded your own images that are not in PNG)
python nft_analysis/convert_images.py   --source_dir data/images_raw   --target_dir data/images

# 1) Extract handcrafted features
python nft_analysis/extract_handcrafted_features.py   --input_metadata data/nft_metadata.parquet   --image_base_dir data/images

# 2) Train traditional ML models
python nft_analysis/train_traditional_models.py   --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv

# 3) Extract deep‑learning features & train models
python nft_analysis/train_deeplearning_models.py   --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv   --image_base_dir data/images

# 4) Collection-level photorelevance analysis
python nft_analysis/correlation_analysis.py   --input_csv outputs/traditional/summaries/selected_features_for_modeling.csv

python nft_analysis/analyze_correlations.py   --input_csv outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv

python nft_analysis/correlation_features.py   --correlations_input_csv outputs/experiment_correlations/feature_fisher_z_correlations.csv   --main_data_input_csv outputs/experiment_correlations/collection_regression_data_with_fisher_z.csv
```

---


## Citation

Please cite our paper when using this code or data:

```bibtex
@inproceedings{MahrousDiPietro2025Aesthetics,
  author    = {Mahrous, Ahmed and Di Pietro, Roberto},
  title     = {Demystifying the Role of Aesthetics in NFT Pricing: Model, Analysis, and Insights},
  booktitle = {Proceedings of the 7th International Conference on Blockchain Computing and Applications (BCCA)},
  year      = {2025},
  address   = {Dubrovnik, Croatia},
}
```

**Code DOI:** https://doi.org/10.5281/zenodo.16415242  
**Data DOI:** https://doi.org/10.5281/zenodo.16414740

---

## License

- **Code:** MIT License (see [LICENSE](LICENSE))  
- **Data on Zenodo (CC BY 4.0):**

---

## Contact

Ahmed Mahrous — ahmed.mahrous@kaust.edu.sa  
