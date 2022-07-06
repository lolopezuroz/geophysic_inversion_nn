from functions.importation import tensorflow as tf

tf.random.set_seed(0)

def nan_mae(y_true, y_pred):
    y_pred = tf.where(tf.math.is_nan(y_true), 0., y_pred)
    y_true = tf.where(tf.math.is_nan(y_true), 0., y_true)
    loss_per_sample = tf.math.abs(y_true - y_pred)
    return tf.math.reduce_sum(loss_per_sample)

def nan_mse(y_true, y_pred):
    y_pred = tf.where(tf.math.is_nan(y_true), 0., y_pred)
    y_true = tf.where(tf.math.is_nan(y_true), 0., y_true)
    loss_per_sample = tf.math.abs(y_true ** .5 - y_pred ** .5) ** 2
    return tf.math.reduce_sum(loss_per_sample)

compiler_functions = {
    "losses": {
        "nan_mae": nan_mae,
        "nan_mse": nan_mse,
        "mae": tf.keras.losses.MeanAbsoluteError(name = "mae"),
        "mse": tf.keras.losses.MeanSquaredError(name = "mse"),
        "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(name = "binary_crossentropy"),
    },
    "metrics": {
        "nan_mae": nan_mae,
        "nan_mse": nan_mse,
        "binary_crossentropy": tf.keras.metrics.BinaryCrossentropy(name = "binary_crossentropy"),
        "binary_accuracy": tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy"),
    },
    "optimizers": {
        "adam": tf.keras.optimizers.Adam
    }
}