"""

"""
#---Imports--------------+
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras import backend as K
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import shutil
import json
import sys
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
sys.path.insert(1, os.path.join(dirname, "src/helper_functions"))

# Local imports
from plotting import plotting

"""
Class regression_CNN. 
A regression convolutional neural network implemented using tensorflow and keras.
The class allows the user to train and compile the model, evaluate on new datasets and producing statistics.
The class makes it easy to save the model for future use, save images of the
network architecture and save the results and statistics of the training and validation.
WARNING: The data preparation should be handled outside of this class.
"""
class regression_CNN():
    def __init__(self, X_train, X_train_cat, y_train, X_test, X_test_cat, y_test, \
            input_shape, signature, learning_rate, model_name = "regression_CNN_1D", epochs=1000):
        """
        Constructor for the regresion_CNN class

        @arguments:
            X_train:     <numpy.ndarray>
            X_train_cat: <>
            y_train:     <tensorflow.python.framework.ops.EagerTensor>
            X_test:      <numpy.ndarray>
            X_test_cat:  <>
            y_test:      <tensorflow.python.framework.ops.EagerTensor>
            input_shape: <tuple> On the form (width, height, channels)
            model_name:  <string> Given name of the model for plots and saved files.
            epochs:      <int> By default 1000, because early stopping is used for regularization
        @returns:
            None
        """
        self.X_train = X_train
        self.X_train_cat = X_train_cat
        self.y_train = y_train
        self.X_test = X_test
        self.X_test_cat = X_test_cat
        self.y_test = y_test
        self.input_shape = input_shape
        self.signature = signature
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.epochs = epochs
        """ 
        indices = np.where(self.X_train_cat == 0)[0]

        self.X_train = np.delete(self.X_train, indices, axis=0)
        self.X_train_cat = np.delete(self.X_train_cat, indices, axis=0)
        self.y_train = np.delete(self.y_train, indices, axis=0)

        indices = np.where(self.X_test_cat == 0)[0]

        self.X_test = np.delete(self.X_test, indices, axis=0)
        self.X_test_cat = np.delete(self.X_test_cat, indices, axis=0)
        self.y_test = np.delete(self.y_test, indices, axis=0)
        """
        self.model = self._create_model()
        self.history = ""

    def _create_model(self, print_sum=True):
        """
        Hidden method, used for creating the CNN model.
        Adds all the layers of the model and finally returns it.
        Prints out the model architecture by default.

        @arguments:
            print_sum <bool> Wether to print out the architecture or not
        @returns:
            model: <keras.engine.sequential.Sequential> The full bCNN model
        """
        # Create the model inputs
        image_input = layers.Input(shape=self.input_shape)
        categorical_input = layers.Input(shape=(1,))
        
        # Create the model
        if self.signature == "z":
            conv1 = layers.Conv1D(32, (3), activation= "relu", input_shape = self.input_shape)(image_input)
            maxpool1 = layers.MaxPooling1D((2))(conv1)
            conv2 = layers.Conv1D(12, (3), activation = "relu")(maxpool1)
            maxpool2 = layers.MaxPooling1D((2))(conv2)
            conv3 = layers.Conv1D(8, (3), activation = "relu")(maxpool2)
            norm1 = layers.BatchNormalization()(conv3)
            maxpool3= layers.MaxPooling1D((2))(norm1)
            flatten = layers.Flatten()(maxpool3)
            concatenated = layers.concatenate([flatten, categorical_input])
            dense1 = layers.Dense(16, activation = "relu")(concatenated)
            #dense2 = layers.Dense(8, activation = "relu")(dense1)
            norm2 = layers.BatchNormalization()(dense1)
            output = layers.Dense(1, activation = "linear")(norm2)

            model = models.Model(inputs=[image_input, categorical_input], outputs=output)

        else: # jet        
            conv1 = layers.Conv1D(10, (3), activation= "relu", input_shape = self.input_shape)(image_input)
            norm1 = layers.BatchNormalization()(conv1)
            maxpool1 = layers.MaxPooling1D((3))(norm1)
            flatten = layers.Flatten()(maxpool1)
            concatenated = layers.concatenate([flatten, categorical_input])
            dense1 = layers.Dense(16, activation = "relu")(concatenated)
            norm2 = layers.BatchNormalization()(dense1)
            output = layers.Dense(1, activation = "linear")(norm2)

            model = models.Model(inputs=[image_input, categorical_input], outputs=output)

        # Print the model architecture and return the model
        if print_sum:
            print(model.summary())

        return model


    def grid_search_lr(self):
        """
        asdfasdf

        @arguments:
            None
        @returns:
            None
        """
        best_score = float('inf')
        best_params = {}
        val_losses = []
        learning_rates = []
        lrs = np.linspace(0.00001, 0.000099, 100).tolist()
        for lr in lrs:
            # Create the model
            model = self._create_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='logcosh', metrics=['mae', 'mape'])

            # Train the model
            history = model.fit([self.X_train, self.X_train_cat], self.y_train, epochs = self.epochs, batch_size=32,
                            validation_data = ([self.X_test, self.X_test_cat], self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                start_from_epoch=50,
                                restore_best_weights=True)])
 
            # Evaluate the model
            val_loss = history.history['val_loss'][-1]

            # Update the best score and best parameters
            if val_loss < best_score:
                best_score = val_loss
                best_params = {'learning_rate': lr}

            val_losses.append(val_loss)
            learning_rates.append(lr)

        print(best_score)
        print(best_params)
        
        plt.figure()
        plt.plot(learning_rates, val_losses, color="navy", linewidth=1.5)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Monojet regressor. Loss versus learning rate")
        plt.savefig("learning_rate.png")

    
        return best_params['learning_rate']

    
    def compile(self):
        """
        Compiles and save an image of the model architecture. 
        The image is saved in /src/ML_1D/model_pngs/
        @arguments:
            None
        @returns:
            None
        """
        # Try plotting the architecture of the network
 
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=self.model_name + '.png',
                show_shapes=False,
                show_dtype=False,
                show_layer_names=False,
                rankdir='TB',
                expand_nested=False,
                dpi=96,
                layer_range=None,
                show_layer_activations=False,
                show_trainable=False
            )
            # Move the png to the correct folder
            dirname_here = os.getcwd()
            shutil.move(dirname_here + "/" + self.model_name + '.png', dirname_here + "/model_pngs/" + self.model_name+'.png') 

        except FileNotFoundError as e:
            print(f"Could not save image of model architecture. Error: {e}")
        
        learning_rate = self.grid_search_lr()
        #learning_rate = 0.0004
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = "mse", metrics = [tf.keras.metrics.RootMeanSquaredError(), \
                tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), \
                tf.keras.metrics.MeanSquaredLogarithmicError(), tf.keras.metrics.CosineSimilarity(), \
                tf.keras.metrics.LogCoshError()])
    

    def train(self, save_model=True):
        """
        Trains the model using early stopping as regularization technique.

        @arguments:
            save_model <bool> Whether to save the model as a .h5 file or not.
        @returns:
            None
        """
        # Trains the model
        self.history = self.model.fit([self.X_train, self.X_train_cat], self.y_train, epochs = 10, batch_size=32,
                            validation_data = ([self.X_test, self.X_test_cat], self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=80, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                start_from_epoch=1,
                                restore_best_weights=True)])
    
        # Save a loadable .h5 file
        if save_model:
            try:
                self.model.save(self.model_name + ".h5")
                # Move the model to the correct folder
                dirname_here = os.getcwd()
                shutil.move(dirname_here + "/" + self.model_name + '.h5', dirname_here + "/saved_models/" + self.model_name + '.h5') 

            except FileNotFoundError as e:
                print(f"Could not save model as .h5 file. Error: {e}")

        dirname_here = os.getcwd()
        plotter = plotting("", "", self.history, self.model_name, dirname_here + "/plots")
        #plotter.loss(cl_or_re="cl", show=True)
        plotter.loss_re(show=True)
 

