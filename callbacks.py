import tensorflow as tf
from tensorflow import keras
from lifelines.utils import concordance_index
from data import prepare_dataset
import numpy as np
from tensorflow.keras import callbacks

class CustomValCallback(callbacks.Callback):
    def __init__(self, val_df, batch_size, best_model_path, test_df):
        super().__init__()
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.best_model_path = best_model_path
        self.best_cindex = -np.inf
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_ds = prepare_dataset(self.val_df, self.batch_size, shuffle=False)
        test_ds = prepare_dataset(self.test_df, self.batch_size, shuffle=False)
        all_logits = []
        all_times = []
        all_events = []

        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0).squeeze()
            all_logits.extend(-preds)
            all_times.extend(labels[:,0].numpy())
            all_events.extend(labels[:,1].numpy())
            
        test_risk = []
        test_times = []
        test_events = []
            
        for images, labels in test_ds:
            test_p = self.model.predict(images, verbose=0).squeeze()
            test_risk.extend(-test_p)
            test_times.extend(labels[:,0].numpy())
            test_events.extend(labels[:,1].numpy())
            
        

        cindex_val = concordance_index(all_times, all_logits, all_events)
        cindex_test = concordance_index(test_times, test_risk, test_events)
        print(f"\nEpoch {epoch+1} Validation C-Index: {cindex_val:.4f} , Test C-Index: {cindex_test:.4f}")
        val_loss = self.model.evaluate(val_ds, verbose=0)
        print(f"   Validation Loss: {val_loss:.4f}")
        
        # Save best by c-index
        if cindex_val > self.best_cindex + 1e-3:
            self.best_cindex = cindex_val
            self.model.save_weights(self.best_model_path)
            print(f"  Validation C-index improved to {self.best_cindex:.4f}. Model saved.")
        '''
        # Save best by val loss
        if val_loss < self.best_loss + 1e-5:
            self.best_loss = val_loss
            self.model.save_weights(self.best_model_path)
            print(f"  Validation Loss improved to {self.best_loss:.4f}. Model saved.")
        '''
        
            
       