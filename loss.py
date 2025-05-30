import tensorflow as tf
from tensorflow import keras


class RankLoss(keras.losses.Loss):
    """
    Pairwise rank loss for keras CNN.
    """
    def __init__(self, name='cox_rank_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        y_true: (batch_size, 2) => [time, event]
        y_pred: (batch_size, 1) => predicted risk
        """
        times = y_true[:, 0]   # shape: (B,)
        events = y_true[:, 1]  # shape: (B,)
        risk = tf.squeeze(y_pred, axis=-1)  # (B,)
        
        times_i = tf.expand_dims(times, axis=1)   # (B,1)
        times_j = tf.expand_dims(times, axis=0)   # (1,B)
        events_i = tf.expand_dims(events, axis=1) # (B,1)
        risk_i = tf.expand_dims(risk, axis=1)     # (B,1)
        risk_j = tf.expand_dims(risk, axis=0)     # (1,B)

        valid_mask = tf.logical_and(tf.equal(events_i, 1.0),
                                    tf.less(times_i, times_j))
        valid_mask_f = tf.cast(valid_mask, dtype=tf.float32)
        
        pairwise_loss = tf.math.log(1.0 + tf.exp(risk_j - risk_i))
        
        loss_for_valid_pairs = pairwise_loss * valid_mask_f 
        
        sampled_losses = tf.reduce_sum(loss_for_valid_pairs)

        if tf.size(sampled_losses) == 0:
            tf.print("WARNING: sampled_losses is empty!")

        loss_sum = tf.reduce_sum(sampled_losses)
        return loss_sum


