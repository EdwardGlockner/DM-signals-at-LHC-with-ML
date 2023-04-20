import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class CNN_model():
    def __init__(self, X_train, y_train, X_test, y_test, epochs=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.model = self._create_model()
        self.history = ""
        self.callback = Callback()


    def _create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation= "relu", input_shape = (28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        # three splits in data
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation = "relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(16, activation = "relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1, activation = "linear"))
        return model


    def compile(self, model_name):
        print(self.model.summary())
        tf.keras.utils.plot_model(
            self.model,
            to_file= model_name + ".png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

        self.model.compile(optimizer = "sgd", loss = "mse", metrics = [tf.keras.metrics.RootMeanSquaredError()])

    
    def train(self, print_perf=True):
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 1000,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=2, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                restore_best_weights=True)])

        self.model.save("./model.h5")
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)


    def plot_performance(self):
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


