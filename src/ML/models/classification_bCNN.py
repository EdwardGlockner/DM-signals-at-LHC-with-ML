"""
Created 24/4-23
"""
#---Imports--------------+
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models
tfd = tfp.distributions
tfpl = tfp.layers

"""

"""
class classification_bCNN():
    def __init__(self, X_train, y_train, X_test, y_test, input_shape, model_name = "classification_bCNN", epochs=1000):
        """
        @arguments:
            X_train:     <numpy.ndarray>
            y_train:     <tensorflow.python.framework.ops.EagerTensor>
            X_test:      <numpy.ndarray>
            y_test:      <tensorflow.python.framework.ops.EagerTensor>
            input_shape: <tuple> On the form: (width, height, channels)
            model_name:  <string> Given name of the model for plots and saved files.
            epochs:      <int> By default 1000, because early stopping is used for regularization
        @returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.input_shape = input_shape
        self.model_name = model_name
        self.epochs = epochs  
        self.model = self._create_model()
        self.history = None

    
    def _create_model(self, print_sum=True):
        """
        Adds all the layers of the model and finally returning it.
        Prints out the model architecture by default.

        @arguments:
            print_sum <bool> Wether to print out the architecture or not
        @returns:
            model: <keras.engine.sequential.Sequential> The full bCNN model
        """
        # Create the model
        model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation= "relu", padding = "VALID", input_shape = self.input_shape),
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
        if print_sum:
            print(model.summary())

        return model

    
    def n_ll(self, y_true, y_pred):
        """
        Calculates the negative log likelihood function. 
        Used as a loss function for the training of bCNN.

        @arguments:
            y_true: <tensorflow.python.framework.ops.Tensor>
            y_pred: <tensorflow_probability.python.layers.internal.distribution_tensor_coercible._TensorCoercible>
        @returns:
            neg_log_lik: <tensorflow.python.framework.ops.Tensor>
        """

        neg_log_lik = -y_pred.log_prob(y_true)

        return neg_log_lik 



    def compile(self):
        """
        Saves an image of the model architecture and compiles it. 

        @arguments:
            None
        @returns:
            None
        """

        tf.keras.utils.plot_model(
        self.model,
        to_file = self.model_name + ".png",
        show_shapes = True,
        show_layer_names = True,
        rankdir = "TB",
        expand_nested = True,
        dpi =96)

        self.model.compile(optimizer = "adam",
                           loss = self.n_ll,
                           metrics = ["accuracy"])


    def train(self, save_model=False):
        """
        Trains the model using early stopping as regularization technique.

        @arguments:
            save_model <bool> Whether to save the model as a .h5 file or not.
        @returns:
            None
        """

        self.history = self.model.fit(self.X_train, self.y_train, epochs = self.epochs,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=1)])
        
        if save_model:
            self.model.save("../saved_models/" + self.model_name + ".h5")


    def plot_performance(self):
        """
        Plots the performance of the model throughout the training set.
        The accuracy is plotted versus the number of epochs, for both the validation
        and the training.

        @arguments:
            None
        @returns:
            None
        """
        plt.plot(self.history.history["accuracy"], label = "accuracy")
        plt.plot(self.history.history["accuracy"], label = "val_accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1])
        plt.legend(loc="lower right")
        plt.show()


    def analyze_model_prediction(self, data, true_label, model, image_num, run_ensamble=False):
        """
        not implemented
        @arguments:
            data:
            true_label:
            model:
            image_num:
            run_ensamble: <bool>
        @returns:
            None
        """
        """
        if run_ensamble:
            ensamble_size = 200
        else:
            ensamble_size = 1

        #image = 
        """
        raise NotImplementedError
    



