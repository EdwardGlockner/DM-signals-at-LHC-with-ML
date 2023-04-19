import tensorflow as tf
import tensorflow_probability as tfp
import keras
import os
import warnings
from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib.backends import backend_agg
import seaborn as sns
import numpy as np
from tensorflow.keras import datasets, layers, models

tfd = tfp.distributions
tfpl = tfp.layers

class bCNN_model():
    def __init__(self, X_train, y_train, X_test, y_test, epochs = 5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = 5
        self.model = self._create_model()

    
    def _create_model(self):
        model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation= "relu", padding = "VALID", input_shape = (28, 28, 1)),
            layers.BatchNormalization(),
            #layers.MaxPooling2D(pool_size = (2,2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu", padding = "VALID"),
            layers.BatchNormalization(),
            #layers.MaxPooling2D(pool_size = (2,2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu", padding = "VALID"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu", padding = "VALID"),
            #layers.MaxPooling2D(pool_size = (2,2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.Dense(tfpl.OneHotCategorical.params_size(10)),
            tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)
        ])

        return model

    
    def n_ll(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)



    def compile(self, model_name):
        print(self.model.summary())
        tf.keras.utils.plot_model(
                self.model,
                to_file = "model.png",
                show_shapes = True,
                show_layer_names = True,
                rankdir = "TB",
                expand_nested = True,
                dpi =96
        )

        self.model.compile(optimizer = "adam",
                           loss = self.n_ll,
                           metrics=[keras.metrics.RootMeanSquaredError(), "accuracy"])


    def train(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 1000,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=False)])

        self.model.save("./model.h5")


    def plot_performance(self):
        pass


    def analyze_model_prediction(self, data, true_label, model, image_num, run_ensamble=False):
        if run_ensamble:
            ensamble_size = 200
        else:
            ensamble_size = 1

        #image = 
