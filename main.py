import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import optimizers, callbacks
from data import *
from model import build_resnet3d
from loss import RankLoss
from callbacks import CustomValCallback
from utils import plot_training_history
from lifelines.utils import concordance_index
from config import *
from explain import shap_explanation, grad_cam_3d, overlay_and_save_gradcam

# Set seeds for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Load and prepare data
print("Loading and splitting data...")
train_df, val_df, test_df = load_data()
train_ds = prepare_dataset(train_df, BATCH_SIZE, shuffle=True)
val_ds = prepare_dataset(val_df, BATCH_SIZE, shuffle=False)
test_ds = prepare_dataset(test_df, BATCH_SIZE, shuffle=False)

# Build and compile model
print("Building and compiling model...")
model = build_resnet3d()
loss_fn = RankLoss()
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=loss_fn)

# Define callbacks
early_stopping = callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=6, verbose=1, min_lr=1e-9)
cindex_callback = CustomValCallback(val_df, BATCH_SIZE, BEST_MODEL_PATH, test_df)

callback_list = [early_stopping, reduce_lr, cindex_callback]

# Train the model
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callback_list,
    verbose=1
)

# Evaluate on test set
print("Reloading best checkpoint...")
best_model = build_resnet3d()
best_model.load_weights(BEST_MODEL)
best_model.compile(optimizer=optimizer, loss=loss_fn)

# Plot training history
print("Plotting training history...")
plot_training_history(history)
print("Done!")

print("Restructing model for feature extraction....")
extractor = tf.keras.Model(inputs=best_model.input, outputs=best_model.get_layer('global_average_pooling3d_1').output)

# Feature extraction
print("Accessing data split information")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Extracting features ....")
train_ds = make_dataset(train_df, BATCH_SIZE)
val_ds   = make_dataset(val_df,   BATCH_SIZE)
test_ds  = make_dataset(test_df,  BATCH_SIZE)
for split_name, ds, df in (
    ('train', train_ds, train_df),
    ('val',   val_ds,   val_df),
    ('test',  test_ds,  test_df)
):
    out = extract_features(extractor, ds, df)
    csv_path = os.path.join(OUTPUT_DIR, f'{split_name}_features.csv')
    out.to_csv(csv_path, index=False)
    print(f"Saved {split_name} features to {csv_path}")
    
# SHAP explanation
try:
    shap_explanation(best_model, train_df, SAMPLE_INDEX, os.path.join(EXPLAIN_DIR, 'train_sample'))
except Exception as e:
    print("SHAP Generation failed. It is advised to reduce SHAP background samples and iterations in the config file \n {e}")

# Grad-CAM visualization
sample_path = train_df['nifti_path'].iloc[SAMPLE_INDEX]
preprocessed_mri = load_and_preprocess_nifti(sample_path, TARGET_SHAPE)
batch_mri = np.expand_dims(preprocessed_mri, 0)
heatmap_3d = grad_cam_3d(best_model, batch_mri, GRAD_CAM_LAYER)
if heatmap_3d is not None:
    overlay_and_save_gradcam(preprocessed_mri, heatmap_3d, os.path.join(OUTPUT_DIR, 'gradcam_3d_output'))
else:
    print("Grad-CAM generation failed.")