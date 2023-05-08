import tensorflow as tf
import shutil
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import sys

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("ml_experiments/CNN/classification","")
sys.path.insert(1, os.path.join(dirname, "src/helper_functions"))
from plotting import plotting

class CNN_model():
    def __init__(self, X_train, y_train, X_test, y_test, epochs=5):
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
        self.epochs = epochs
        self.model = self._create_model()
        self.history = ""
        self.callback = Callback()


    def _create_model(self):
        """
        asdfasdf

        @arguments:
            None

        @returns:
            Model: 

        """
        # Add all the layers
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation="softmax"))

        print(model.summary())
        return model

    def compile(self, model_name):
        """
        asdfasdf

        @arguments:
            model_name: <string> Name of the figure that will be saved

        @returns:
            None
        """
        # Save figure of the model
        tf.keras.utils.plot_model(
            self.model,
            to_file= model_name + ".png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )
        
        # Compile the model
        self.model.compile(optimizer = "adam",
                      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics = ["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), \
                              tf.keras.metrics.Recall(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), \
                              tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    def evaluate_model(self, X_val, y_val, save_stats=True):
        """
        asdfasdf

        @arguments:
            X_val: <>
            y_val: <>
            save_stats: <>
        @returns:

        """
        try:
            results = self.model.evaluate(X_val, y_val, batch_size=128)
        except OverflowError as e:
            print(f"Error occured in <evaluate_model>. Error: {e}")
            return None

        predictions = self.model.predict(X_val[:])
        stats = {
            'loss': results[0],
            'accuracy': results[1],
            'AUC': results[2],
            'precision': results[3],
            'recall': results[4],
            'TP': results[5],
            'TN': results[6],
            'FP': results[7],
            'FN': results[8],
            'prediction': predictions.tolist(),
            'y_test': y_val.tolist()
        }

        if save_stats:
            with open("test" + '.json', 'w') as f:
                json.dump(stats, f)
            dirname_here = os.getcwd()
            try:
                shutil.move(dirname_here + "/" + "test" + ".json", dirname_here + "/val_stats/" + "test" + ".json") 
            except FileNotFoundError as e:
                print(f"Could not save validation statistics. Error: {e}")
        
        model_name = "test_CNN"
        plotter = plotting(self.y_test, predictions, self.history, model_name, "/Users/edwardglockner/OneDrive - Uppsala universitet/Teknisk Fysik/Termin 6 VT23/Kandidatarbete/DM-signals-at-LHC-with-ML/ml_experiments/CNN/classification")
        plotter.loss(cl_or_re="cl", show=True)
        plotter.accuracy(show=True)
        plotter.roc(num_classes = 10, show=True)

    def train(self, print_perf=True):
        """
        asdfasdf

        @arguments:
            print_perf: <bool> 

        @returns:
            None
        """
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 3,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                start_from_epoch=1,
                                restore_best_weights=True)])

        self.model.save("./model.h5")
        self.evaluate_model(self.X_test, self.y_test)

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


