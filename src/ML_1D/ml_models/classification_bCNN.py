"""
Created 24/4-23
"""
#---Imports--------------+
import tensorflow as tf
import sys
import os
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models
tfd = tfp.distributions
tfpl = tfp.layers
import os
import shutil
import json

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
sys.path.insert(1, os.path.join(dirname, "src/helper_functions"))

from plotting import plotting


"""

"""
class classification_bCNN():
    def __init__(self, X_train, y_train, X_test, y_test, input_shape, num_classes, model_name = "classification_bCNN_1D", epochs=1000):
        """
        @arguments:
            X_train:     <numpy.ndarray>
            y_train:     <tensorflow.python.framework.ops.EagerTensor>
            X_test:      <numpy.ndarray>
            y_test:      <tensorflow.python.framework.ops.EagerTensor>
            input_shape: <tuple> On the form: (width, height, channels)
            num_classes: <int> Numer of classification labels
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
        self.num_classes = num_classes
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
            layers.Conv1D(filters=32, kernel_size=(3), activation= "relu", padding = "VALID", input_shape = self.input_shape),
            layers.MaxPooling1D(pool_size = (2)),
            layers.Conv1D(filters=32, kernel_size=(3), activation = "relu", padding = "VALID"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size = (2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(tfpl.OneHotCategorical.params_size(self.num_classes)),
            tfpl.OneHotCategorical(self.num_classes, convert_to_tensor_fn=tfd.Distribution.mode)
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

        self.model.compile(optimizer = "adam",
                           loss = self.n_ll,
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
                shutil.move(dirname_here + "/" + "test" + ".json", \
                        dirname_here + "/val_stats/" + self.model_name + ".json") 
            except FileNotFoundError as e:
                print(f"Could not save validation statistics. Error: {e}")
        
        plotter = plotting(self.y_test, predictions, self.history, self.model_name, "/Users/edwardglockner/OneDrive - Uppsala universitet/Teknisk Fysik/Termin 6 VT23/Kandidatarbete/DM-signals-at-LHC-with-ML/ml_experiments/CNN/classification")
        plotter.loss(cl_or_re="cl", show=True)
        plotter.accuracy(show=True)
        plotter.roc(num_classes = 10, show=True)


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
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=1)])
        
        if save_model:
            try:
                self.model.save(self.model_name + ".h5")
                # Move the model to the correct folder
                dirname_here = os.getcwd()
                shutil.move(dirname_here + "/" + self.model_name + '.h5', dirname_here + "/saved_models/" + self.model_name + '.h5') 

            except FileNotFoundError as e:
                print(f"Could not save model as .h5 file. Error: {e}")

        
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
    
