import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import shutil

class regression_CNN():
    def __init__(self, X_train, y_train, X_test, y_test, input_shape, model_name = "regression_CNN_2D", epochs=1000):
        """
        asdfasdf

        @arguments:
            X_train:     <numpy.ndarray>
            y_train:     <tensorflow.python.framework.ops.EagerTensor>
            X_test:      <numpy.ndarray>
            y_test:      <tensorflow.python.framework.ops.EagerTensor>
            input_shape: <tuple> On the form (width, height, channels)
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
        self.history = ""
        self.callback = Callback()


    def _create_model(self):
        """
        asfasdf

        @arguments:
            None
        @returns:
            None
        """
        # Add all the layers
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation= "relu", input_shape = self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation = "relu"))
        model.add(layers.Dense(16, activation = "relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1, activation = "linear"))
        
        # Print the model architecture and return the model
        print(model.summary())

        return model


    def compile(self):
        """
        asdfasdf

        @arguments:
            None
        @returns:
            None
        """
        
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


        self.model.compile(optimizer = "sgd", loss = "mse", metrics = [tf.keras.metrics.RootMeanSquaredError()])

    
    def train(self, save_model=True):
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
                                patience=2, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                start_from_epoch=5,
                                restore_best_weights=True)])
        if save_model:
            try:
                self.model.save(self.model_name + ".h5")
                # Move the model to the correct folder
                dirname_here = os.getcwd()
                shutil.move(dirname_here + "/" + self.model_name + '.h5', dirname_here + "/saved_models/" + self.model_name + '.h5') 

            except FileNotFoundError as e:
                print(f"Could not save model as .h5 file. Error: {e}")

        """
        # FOR TESTING
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
        print("test loss, test acc:", results)

        print("Generate predictions for 3 samples")
        predictions = self.model.predict(self.X_test[:])
        print("predictions shape:", predictions.shape)
        print(f"Predictions: {predictions}")
        """

    def plot_performance(self):
        """
        asdfasdf

        @arguments:
            None
        @returns:
            None
        """
        plt.plot(self.history.history["root_mean_squared_error"], label = "MSE")
        plt.plot(self.history.history["val_root_mean_squared_error"], label = "val_MSE")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.ylim([0, 1])
        plt.legend(loc="lower right")
        plt.show()
   

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


