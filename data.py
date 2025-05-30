import os
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import *

def load_and_preprocess_nifti(file_path, target_shape=TARGET_SHAPE):
    """Load a NIfTI file, resize it, and normalize it."""
    if isinstance(file_path, bytes):
        file_path = file_path.decode('utf-8')
    try:
        nii = nib.load(file_path)
        volume = nii.get_fdata(dtype=np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(target_shape, dtype=np.float32)

    # Resize if necessary
    if volume.shape != tuple(target_shape):
        zoom_factors = [t / o for t, o in zip(target_shape, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)

    # Z-score normalization
    mean, std = np.mean(volume), np.std(volume)
    std = max(std, 1e-5)  # Avoid division by zero
    volume = (volume - mean) / std
    return volume[..., np.newaxis].astype(np.float32)

def prepare_dataset(df, batch_size, shuffle=False):
    """Create a tf.data.Dataset from a dataframe."""
    file_paths = df['nifti_path'].values
    times = df['time'].values
    events = df['convert_status'].values

    def generator():
        for fp, t, e in zip(file_paths, times, events):
            vol = load_and_preprocess_nifti(fp)
            yield vol, np.array([t, e], dtype=np.float32)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(*TARGET_SHAPE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(32)
    return ds

def load_data():
    """Load and split the dataset."""
    df = pd.read_csv(DATA_DTA)
    if 'nifti_path' not in df.columns:
        df['nifti_path'] = df['imageid'].apply(lambda x: os.path.join(MRI_FOLDER, f'{x}.nii'))
        
    # Filter out missing files
    df = df[df['nifti_path'].apply(os.path.exists)]
    df = df[['subjectid', 'nifti_path', 'convert_status', 'time', 'imageid']].drop_duplicates(subset=['subjectid'])

    # Split data
    trainval_df, test_df = train_test_split(df, test_size=TEST_RATIO, stratify=df['convert_status'], random_state=SEED)
    train_df, val_df = train_test_split(trainval_df, test_size=VAL_TRAIN_RATIO, stratify=trainval_df['convert_status'], random_state=SEED)

    # Save splits
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_SET_CSV, index=False)
    return train_df, val_df, test_df
    

def build_nifti_paths(df):
    """Add NIfTI paths to dataframe if missing."""
    if 'nifti_path' not in df.columns:
        df['nifti_path'] = df['imageid'].apply(lambda x: os.path.join(MRI_FOLDER, f'{x}.nii'))
    return df

def make_dataset(df, batch_size):
    """Create a TensorFlow dataset from a dataframe."""
    df = build_nifti_paths(df)
    paths = df['nifti_path'].values
    times = df['time'].values.astype(np.float32)
    events = df['convert_status'].values.astype(np.float32)
    imageids = df['imageid'].values

    ds = tf.data.Dataset.from_tensor_slices((paths, times, events, imageids))
    def _map(path, t, e, img_id):
        vol = tf.numpy_function(load_and_preprocess_nifti, [path, TARGET_SHAPE], tf.float32)
        vol.set_shape((*TARGET_SHAPE, 1))
        return vol, (t, e, img_id)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def extract_features(extractor, ds, df):
    features_list = []
    ids = []
    times = []
    events = []
    for batch_vols, (batch_t, batch_e, batch_img) in ds:
        feats = extractor.predict(batch_vols, verbose=0)
        features_list.append(feats)
        ids.extend(batch_img.numpy())
        times.extend(batch_t.numpy())
        events.extend(batch_e.numpy())
    feats_array = np.vstack(features_list)
    # build DataFrame
    out_df = pd.DataFrame({
        'subjectid': df['subjectid'].values,
        'imageid': df['imageid'].values,
        'time': times,
        'convert_status': events
    })
    # add feature columns
    for i in range(feats_array.shape[1]):
        out_df[f'feature_{i+1}'] = feats_array[:, i]
    return out_df
