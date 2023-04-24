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
        """
        asdfasdf

        @arguments:
            X_train: <numpy.ndarray>
            y_train: <tensorflow.python.framework.ops.EagerTensor>
            X_test:  <numpy.ndarray>
            y_test:  <tensorflow.python.framework.ops.EagerTensor>
            epochs:  <int>

        @returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = 5
        self.model = self._create_model()

    
    def _create_model(self):
        """
        asdfasdf

        @arguments:
            None

        @returns:
            None
        """
        # Create the model
        model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation= "relu", padding = "VALID", input_shape = (28, 28, 1)),
            layers.MaxPooling2D(pool_size = (2, 2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu", padding = "VALID"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size = (2,2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(tfpl.OneHotCategorical.params_size(10)),
            tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)
        ])
        # Print the architecture and return the model
        print(model.summary())

        return model

    
    def n_ll(self, y_true, y_pred):
        """
        asdfasdf

        @arguments:
            y_true: <tensorflow.python.framework.ops.Tensor>
            y_pred: <tensorflow_probability.python.layers.internal.distribution_tensor_coercible._TensorCoercible>

        @returns:
            neg_log_lik: <tensorflow.python.framework.ops.Tensor>
        """
        neg_log_lik = -y_pred.log_prob(y_true)

        return neg_log_lik 



    def compile(self, model_name):
        """
        asdfsdf

        @arguments:
            model_name: <string> Name of the model that will be saved

        @returns:
            None
        """
        tf.keras.utils.plot_model(
        self.model,
        to_file = "model.png",
        show_shapes = True,
        show_layer_names = True,
        rankdir = "TB",
        expand_nested = True,
        dpi =96)

        self.model.compile(optimizer = "adam",
                           loss = self.n_ll,
                           #metrics=[keras.metrics.RootMeanSquaredError(), "accuracy"])
                           metrics = ["accuracy"])


    def train(self):
        """
        asdfasdf

        @arguments:
            None
        @returns:
            None

        """
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 1000,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=1)])

        self.model.save("./model.h5")


    def plot_performance(self):
        """
        asdfasdf

        @arguments:
            None

        @returns:
            None
        """
        pass


    def analyze_model_prediction(self, data, true_label, model, image_num, run_ensamble=False):
        """
        aasdfasdf

        @arguments:
            data:
            true_label:
            model:
            image_num:
            run_ensamble: <bool>

        @returns:
            None
        """
        if run_ensamble:
            ensamble_size = 200
        else:
            ensamble_size = 1

        #image = 
