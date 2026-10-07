"""MLP (Keras) para a transição de categoria t0 -> t1."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


def build(n_features, cfg, n_classes=6):
    import tensorflow as tf
    m = cfg["model"]; tf.keras.utils.set_random_seed(m["seed"])
    layers = [tf.keras.layers.Input((n_features,))]
    for h in m["hidden"]:
        layers += [tf.keras.layers.Dense(h, activation="relu"), tf.keras.layers.Dropout(m["dropout"])]
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(m["lr"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, Xtr, ytr, Xva, yva, cfg):
    import tensorflow as tf
    m = cfg["model"]
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=m["patience"], restore_best_weights=True)
    return model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=m["max_epochs"], batch_size=m["batch"],
                     callbacks=[es], verbose=0)
