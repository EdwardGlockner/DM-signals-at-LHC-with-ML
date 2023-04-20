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
from tensorflow.keras import regularizers
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
        model = tf.keras.Sequential([
            tfpl.Convolution2DFlipout(32, kernel_size=(3, 3), activation="relu", input_shape = (28, 28, 1)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation= "relu", padding = "VALID"),
            layers.MaxPooling2D(pool_size = (2,2)),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu", padding = "VALID"),
            layers.MaxPooling2D(pool_size = (2,2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.5),
            #tfpl.DenseFlipout(128, activation="relu"),
            #layers.BatchNormalization(),
            #layers.Dropout(0.5),
            #tfpl.DenseFlipout(64, activation="relu"),
            #layers.BatchNormalization(),
            #tfpl.DenseFlipout(1)
            

            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(16, activation="relu"),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
            


            #tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1/self.X_train.shape[0]),
            #tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))

            ])


        return model

    
    def n_ll(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
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
        neg_log_likelihood = lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        self.model.compile(optimizer="sgd", metrics = ["mse", "mae"], loss=neg_log_likelihood)

    def train(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 1000,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=2, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=True)])

        self.model.save("./model.h5")


    def plot_performance(self):
        pass


    def analyze_model_prediction(self, data, true_label, model, image_num, run_ensamble=False):
        if run_ensamble:
            ensamble_size = 200
        else:
            ensamble_size = 1

        #image = 



# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])
