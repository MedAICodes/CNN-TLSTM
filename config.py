import os
from pathlib import Path

# Paths (update these to your local setup)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DTA = ROOT_DIR / "data" / "datafile.csv" # A csv table containing subjectid, imageid, and time to event outcome (convert_status, time) columns, optionally path to each preprocessed MRI file can be given via nifti_path column 
MRI_FOLDER = ROOT_DIR / "data" / "MRI" # Folder containing MRI files, each named as imageid + .nii
TRAIN_CSV = ROOT_DIR / "data" / "train_set.csv" # Directory for storing data splits / Extracting features for subsequent longitudinal training
VAL_CSV = ROOT_DIR / "data" / "val_set.csv" # Table for storing data splits IDs / Extracting features from IDs for subsequent longitudinal training
TEST_SET_CSV = ROOT_DIR / "data" / "test_set.csv" # Table for storing data splits IDs / Extracting features from IDs for subsequent longitudinal training
BEST_MODEL_PATH = ROOT_DIR / "models" / "best_model_cox.h5" # Best performing model saving location

TEST_CSV = TEST_SET_CSV
BEST_MODEL = BEST_MODEL_PATH
OUTPUT_DIR = ROOT_DIR / "outputs" # Directory for extracted features
EXPLAIN_DIR = OUTPUT_DIR / "model_explanations" # Directory for 3D GRADCAM and SHAP images
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)

TEST_RATIO = 0.175 # Test/Total size
VAL_TRAIN_RATIO = 0.15 # Validation/Training size

# Hyperparameters
BATCH_SIZE = 16
DROPOUT_RATE = 0.45
EPOCHS = 100
TARGET_SHAPE = [128, 128, 128]
SEED = 77777
LEARNING_RATE = 1e-4

# Explanation settings
SHAP_BACKGROUND_SIZE = 8 
SHAP_NSAMPLES = 10        
SHAP_LOW_PERCENTILE_CUT = 0.05 # Outlier clipping for Shap 
SHAP_HIGH_PERCENTILE_CUT = 99.95 # Outlier clipping for Shap 
GRAD_CAM_LAYER = "conv3d_39"  # Changes upon each iteration of model reloading, also check global pooling layer name via main.py
SAMPLE_INDEX = 10          # Index of a training set sample to generate GRADCAM and SHAP explanations