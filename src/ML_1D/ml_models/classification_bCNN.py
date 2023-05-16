"""
Created 24/4-23
"""
#---Imports--------------+
import tensorflow as tf
import sys
import os
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models
tfd = tfp.distributions
tfpl = tfp.layers
import os
import shutil
import json
from keras.utils import to_categorical

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
sys.path.insert(1, os.path.join(dirname, "src/helper_functions"))

# Local imports
from plotting import plotting

"""
Class classification_bCNN. 
A classification bayesian convolutional neural network implemented using tensorflow and tensorflow_probability.
The class allows the user to train and compile the model, evaluate on new datasets and producing statistics.
The class makes it easy to save the model for future use, save images of the
network architecture and save the results and statistics of the training and validation.
WARNING: The data preparation should be handled outside of this class, including
normalization, shuffling and so on.
"""

class classification_bCNN():
    def __init__(self, X_train, y_train, X_test, y_test, input_shape, num_classes, model_name = "classification_bCNN_1D", epochs=1000):
        """
        Constructor for the classification_bCNN class

        @arguments:
            X_train:     <numpy.ndarray> Arbitrary input data set used for training 
            y_train:     <tensorflow::EagerTensor> Output labels for the training set
            X_test:      <numpy.ndarray> Arbitrary input data used for testing
            y_test:      <tensorflow::EagerTensor> Output labels for the testing set
            input_shape: <tuple> Shape of the input data sets. On the form: (width, height, channels)
            num_classes: <int> Numer of classification labels
            model_name:  <string> Given name of the model for plots and saved files.
            epochs:      <int> Maximum numbers of epochs. By default 1000 since early stopping is used for regularization
        @returns:
            None
        """
        self.X_train = X_train
        self.y_train = to_categorical(y_train, num_classes)
        self.X_test = X_test
        self.y_test = to_categorical(y_test, num_classes)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.epochs = epochs  
        self.model = self._create_model()
        self.history = None

    
    def _create_model(self, print_sum=True):
        """
        Hidden method, used for creating the bCNN model.
        Adds all the layers of the model and finally returns it.
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
            y_true: <tensorflow::Tensor> The actual output labels for a given set.
            y_pred: <tensorflow_probability::_TensorCoercible> The predicted output labels for the same set.
        @returns:
            neg_log_lik: <tensorflow::Tensor> The negative log likelihood.
        """

        neg_log_lik = -y_pred.log_prob(y_true)

        return neg_log_lik 


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
        
        # Compiles the model
        self.model.compile(optimizer = "adam",
                           loss = self.n_ll,
                            metrics = ["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), \
                              tf.keras.metrics.Recall(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), \
                              tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])


    def evaluate_model(self, X_val, y_val, save_stats=True):
        """
        Evaluates the compiled and trained model on a new validation set.
        Creates all the different performance metrics and saves it to a .json file,
        in src/ML_1D/val_stats/
        Also creates plots of the ROC curves and accuracy, which is saved to
        src/ML_1D/

        @arguments:
            X_val: <numpy.ndarray> Arbitrary input data used for validation. Same input shape as the trained model is needed.
            y_val: <tensorflow::EagerTensor> Output labels for the validation data
            save_stats: <bool> Whether to save the .json file or not.
        @returns:
            None
        """
        y_val_encode = to_categorical(y_val, self.num_classes)
        try:
            results = self.model.evaluate(X_val, y_val_encode, batch_size=128)
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
        plotter = plotting(y_val_encode, predictions, self.history, self.model_name, dirname_here + "/plots")
        plotter.loss(cl_or_re="cl", show=True)
        plotter.accuracy(show=True)
        plotter.roc(num_classes = self.num_classes, show=True)


    def train(self, save_model=True):
        """
        Trains the model using early stopping as regularization technique.

        @arguments:
            save_model <bool> Whether to save the model as a .h5 file or not.
        @returns:
            None
        """
        # Trains the model
        self.history = self.model.fit(self.X_train, self.y_train, epochs = self.epochs,
                            validation_data = (self.X_test, self.y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=1, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=1)])

        # Save a loadable .h5 file   
        if save_model:
            try:
                self.model.save(self.model_name + ".h5")
                # Move the model to the correct folder
                dirname_here = os.getcwd()
                shutil.move(dirname_here + "/" + self.model_name + '.h5', dirname_here + "/saved_models/" + self.model_name + '.h5') 

            except FileNotFoundError as e:
                print(f"Could not save model as .h5 file. Error: {e}")

        
    def analyze_model_prediction(self, data, true_label, model, image_num, run_ensamble=False):
        raise NotImplementedError
    
