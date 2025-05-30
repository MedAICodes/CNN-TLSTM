import matplotlib.pyplot as plt
import os
from config import OUTPUT_DIR

def plot_training_history(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()