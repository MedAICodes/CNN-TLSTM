import shap
import cv2
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import zoom
import tensorflow as tf
import numpy as np
from config import *
from data import load_and_preprocess_nifti

def shap_explanation(model, train_df, sample_index, output_dir):
    """Generate SHAP explanations for a sample."""
    bg_paths = train_df['nifti_path'].sample(n=SHAP_BACKGROUND_SIZE, random_state=SEED).values
    background = np.stack([load_and_preprocess_nifti(p, TARGET_SHAPE) for p in bg_paths], axis=0)
    explainer = shap.GradientExplainer(model, background)
    sample_path = train_df['nifti_path'].iloc[sample_index]
    sample_arr = load_and_preprocess_nifti(sample_path, TARGET_SHAPE)
    sample_batch = np.expand_dims(sample_arr, 0)
    shap_vals = explainer.shap_values(sample_batch, nsamples=SHAP_NSAMPLES)
    shap_map = np.squeeze(shap_vals[0] if isinstance(shap_vals, list) else shap_vals)
    overlay_and_save_shap(sample_arr, shap_map, output_dir)

def overlay_and_save_shap(pre, heatmap, out_dir, save_slices=True):
    """Overlay SHAP heatmap on MRI and save slices/GIF."""
    os.makedirs(out_dir, exist_ok=True)
    heatmap_3d = np.squeeze(heatmap)
    pre_3d = np.squeeze(pre)
    p_low, p_high = np.percentile(heatmap_3d, (SHAP_LOW_PERCENTILE_CUT, SHAP_HIGH_PERCENTILE_CUT))
    heatmap_3d = np.clip(heatmap_3d, p_low, p_high)
    cmap = LinearSegmentedColormap.from_list(
        'blue_transparent_red',
        [(0.0, 'blue'), (0.5, (1.0, 1.0, 1.0, 0.0)), (1.0, 'red')]
    )
    try:
        norm = TwoSlopeNorm(vmin=p_low, vcenter=0.0, vmax=p_high)
    except Exception as e:
        print(f"Single sided SHAP values, Results are unstable: {e}")
        norm = TwoSlopeNorm(vcenter=0.0)
    
    overlaid_slices = []
    for i in range(heatmap_3d.shape[1]):
        slice_pre = pre_3d[:, i, :]
        slice_hm = heatmap_3d[:, i, :]
        slice_norm = (slice_pre - slice_pre.min()) / (slice_pre.max() - slice_pre.min() + 1e-5)
        gray = np.stack([slice_norm] * 3, axis=-1)
        rgba = cmap(norm(slice_hm))
        alpha = rgba[..., 3][..., None]
        overlay = rgba[..., :3] * alpha + gray * (1 - alpha)
        overlay = np.rot90(overlay, k=1)
        overlaid_slices.append(overlay)
        if save_slices:
            plt.imsave(os.path.join(out_dir, f'slice_{i:03d}.png'), overlay)
    gif_path = os.path.join(out_dir, 'shap_3d.gif')
    frames = [(np.clip(sl, 0, 1) * 255).astype(np.uint8) for sl in overlaid_slices]
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"SHAP overlays saved in {out_dir} (GIF: {gif_path})")

def grad_cam_3d(model, img_array, layer_name):
    """Generate Grad-CAM heatmap for a 3D volume."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        print("Gradient is None. Check layer name or model structure.")
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2, 3))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[..., i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 1e-5:
        heatmap /= max_val
    if heatmap.shape != TARGET_SHAPE:
        zoom_factors = [t / o for t, o in zip(TARGET_SHAPE, heatmap.shape)]
        heatmap = zoom(heatmap, zoom_factors, order=1)
    return heatmap

def overlay_and_save_gradcam(preprocessed_mri, heatmap_3d, output_folder, save_slices=True):
    """Overlay Grad-CAM heatmap on preprocessed MRI and save."""
    os.makedirs(output_folder, exist_ok=True)
    pre_3d = np.squeeze(preprocessed_mri)
    heatmap_uint8 = (np.clip(heatmap_3d, 0, 1) * 255).astype(np.uint8)
    overlaid_slices = []
    for i in range(pre_3d.shape[1]):
        mri_slice = pre_3d[:, i, :]
        slice_min, slice_max = mri_slice.min(), mri_slice.max()
        mri_slice_norm = (mri_slice - slice_min) / max(slice_max - slice_min, 1e-5)
        mri_slice_uint8 = (mri_slice_norm * 255).astype(np.uint8)
        hm_slice = heatmap_uint8[:, i, :]
        heatmap_color = cv2.applyColorMap(hm_slice, cv2.COLORMAP_JET)
        mri_slice_bgr = cv2.cvtColor(mri_slice_uint8, cv2.COLOR_GRAY2BGR)
        overlay_bgr = cv2.addWeighted(heatmap_color, 0.5, mri_slice_bgr, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = np.rot90(overlay_rgb, k=1)
        overlaid_slices.append(overlay_rgb)
        if save_slices:
            plt.imsave(os.path.join(output_folder, f'slice_{i:03d}.png'), overlay_rgb)
    gif_path = os.path.join(output_folder, 'gradcam_3d.gif')
    imageio.mimsave(gif_path, overlaid_slices, duration=0.1)
    print(f"Grad-CAM overlays saved in {output_folder} (GIF: {gif_path})")
