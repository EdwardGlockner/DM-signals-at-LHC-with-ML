"""

"""
#---Imports--------------+
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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
    def __init__(self, X_train, X_train_cat, y_train, X_test, X_test_cat, y_test, input_shape, model_name = "regression_CNN_1D", epochs=1000):
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
        conv1 = layers.Conv1D(32, (3), activation= "relu", input_shape = self.input_shape)(image_input)
        maxpool1 = layers.MaxPooling1D((2))(conv1)
        conv2 = layers.Conv1D(32, (3), activation = "relu")(maxpool1)
        norm1 = layers.BatchNormalization()(conv2)
        maxpool2 = layers.MaxPooling1D((2))(norm1)
        flatten = layers.Flatten()(maxpool2)
        concatenated = layers.concatenate([flatten, categorical_input])
        dense1 = layers.Dense(64, activation = "relu")(concatenated)
        dense2 = layers.Dense(16, activation = "relu")(dense1)
        norm2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(1, activation = "linear")(norm2)

        model = models.Model(inputs=[image_input, categorical_input], outputs=output)
        
        # Print the model architecture and return the model
        if print_sum:
            print(model.summary())

        return model


    def grid_search_lr(self):
        """

        """
        """
        def build_model(learning_rate):
            model = self._create_model(print_sum=False)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])

            return model
        
        print(self.X_train.shape)  # Should match the number of samples in self.y_train
        print(self.X_train_cat.shape)  # Should match the number of samples in self.y_train
        print(self.y_train.shape)
        lrs = np.linspace(0.005, 0.4, 20).tolist()
        param_grid = {'learning_rate': lrs} 
        model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, error_score='raise')
        grid_search.fit(np.array([self.X_train, self.X_train_cat]), np.array(self.y_train))
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(best_params)
        print(best_score)
        """
        best_score = float('inf')
        best_params = {}

        lrs = np.linspace(0.005, 0.25, 20).tolist()

        for lr in lrs:
            # Create the model
            model = self._create_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

            # Train the model
            history = model.fit([self.X_train, self.X_train_cat], self.y_train, \
                    epochs=10, validation_data=([self.X_test, self.X_test_cat], \
                    self.y_test))

            # Evaluate the model
            val_loss = history.history['val_loss'][-1]

            # Update the best score and best parameters
            if val_loss < best_score:
                best_score = val_loss
                best_params = {'learning_rate': lr}
        print(best_score)
        print(best_params)
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
                show_shapes=True,
                show_dtype=False,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96,
                layer_range=None,
                show_layer_activations=True,
                show_trainable=False
            )
            # Move the png to the correct folder
            dirname_here = os.getcwd()
            shutil.move(dirname_here + "/" + self.model_name + '.png', dirname_here + "/model_pngs/" + self.model_name+'.png') 

        except FileNotFoundError as e:
            print(f"Could not save image of model architecture. Error: {e}")
        
        #learning_rate = self.grid_search_lr()
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.2), loss = "mse", metrics = [tf.keras.metrics.RootMeanSquaredError(), \
                tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), \
                tf.keras.metrics.MeanSquaredLogarithmicError(), tf.keras.metrics.CosineSimilarity(), \
                tf.keras.metrics.LogCoshError()])
    

    def evaluate_model(self, X_val, X_val_cat, y_val, save_stats=True):
        """
        Evaluates the compiled and trained model on a new validation set.
        Creates all the different performance metrics and saves it to a .json file,
        in src/ML_1D/val_stats/

        @arguments:
            X_val: <numpy.ndarray> Arbitrary input data used for validation. Same input shape as the trained model is needed.
            y_val: <tensorflow::EagerTensor> Output labels for the validation data
            save_stats: <bool> Whether to save the .json file or not.
        @returns:
            None
        """
        """
        indices = np.where(X_val_cat == 0)[0]

        X_val = np.delete(X_val, indices, axis=0)
        X_val_cat = np.delete(X_val_cat, indices, axis=0)
        y_val = np.delete(y_val, indices, axis=0)
        """
        try:
            results = self.model.evaluate([X_val, X_val_cat], y_val, batch_size=128)
        except OverflowError as e:
            print(f"Error occured in <evaluate_model>. Error: {e}")
            return None

        predictions = self.model.predict([X_val[:], X_val_cat[:]])
        stats = {
            'loss': results[0],
            'RMSE': results[1],
            'MAE': results[2],
            'MAPE': results[3],
            'MSLE': results[4],
            'CosSim': results[5],
            'LogCoshE': results[6],
            'prediction': predictions.tolist(),
            'y_test': y_val.tolist()
        }

        dirname_here = os.getcwd()
        if save_stats:
            with open(self.model_name + "_val_data" + '.json', 'w') as f:
                json.dump(stats, f)
            try:
                shutil.move(dirname_here + "/" + self.model_name + "_val_data" + ".json", \
                        dirname_here + "/val_stats/" + self.model_name + ".json") 
            except FileNotFoundError as e:
                print(f"Could not save validation statistics. Error: {e}")
       
        # Create the plotting object and create all the plots
        plotter = plotting(self.y_test, predictions, self.history, self.model_name, dirname_here + "/plots")
        plotter.loss(cl_or_re="cl", show=True)
        plotter.rmse(show=True)


    def train(self, save_model=True):
        """
        Trains the model using early stopping as regularization technique.

        @arguments:
            save_model <bool> Whether to save the model as a .h5 file or not.
        @returns:
            None
        """
        # Trains the model
        self.history = self.model.fit([self.X_train, self.X_train_cat], self.y_train, epochs = self.epochs,
                            validation_data = ([self.X_test, self.X_test_cat], self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=2, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                start_from_epoch=20,
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


