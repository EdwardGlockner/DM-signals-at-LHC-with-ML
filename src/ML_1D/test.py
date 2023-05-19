# main.py

#---IMPORTS--------------+
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore tensorflow warning
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
import json

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
# Get the project root directory by going up two levels
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root directory to the module search path
sys.path.append(project_root)

#---LOCAL IMPORTS--------+
from data_prep_1D.data_prep_test import load_sets 
from helper_functions.plotting import *

#---GLOBALS--------------+
np.set_printoptions(threshold=np.inf)

try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS, Linux
        clear = lambda: os.system("clear")

    elif sys.platform in ["win32", "win64"]: #windows
        clear = lambda: os.system("cls")
    
    else:
        clear = lambda: None

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"


#---FUNCTIONS------------+
def load_tf_model(file_path):
    """
    asdfasdf

    @arguments:
        file_path: <string>
    @returns:
        model: <>
    """
    model = load_model(file_path)
    return model


def load_dataset(signature, model_type):
    """
    asdfasdf

    @arguments:
        model_type: <string>
    @returns:
        datasets
    pass
    """
    dataset = load_sets(signature, model_type)
    return dataset

def evaluate(model, dataset, num_classes, model_name):
    """
    asdfasdf

    @arguments:
        model: <>
        dataset: <>
    @returns:
        None
    """
    if len(dataset) == 2: # classification
        model_type = "cl"
        X_test, y_test = dataset
        print(y_test)
        y_test_encode = to_categorical(y_test, num_classes)

    elif len(dataset) == 3: # regression
        model_type = "re"
        X_test_hist, X_test_cat, y_test = dataset

    else:
        return

    if model_type == "cl":
        try:
            results = model.evaluate(X_test, y_test_encode)
        except OverflowError as e:
            print(f"Error occured in <evaluate>. Error: {e}")
            return

        predictions = model.predict(X_test[:])

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
            'y_test': y_test.tolist()
        }
        plotter = plotting(y_test_encode, predictions, None, model_name, dirname + "/src/ML_1D/testing_loadable_models/")
        plotter.roc(num_classes = num_classes, show=True)

    elif model_type == "re":
        try:
            results = model.evaluate([X_test_hist, X_test_cat], y_test, batch_size=128)

        except OverflowError as e:
            print(f"Error occured in <evaluate_model>. Error: {e}")
            return None

        predictions = model.predict([X_test_hist[:], X_test_cat[:]])
        stats = {
            'loss': results[0],
            'RMSE': results[1],
            'MAE': results[2],
            'MAPE': results[3],
            'MSLE': results[4],
            'CosSim': results[5],
            'LogCoshE': results[6],
            'prediction': predictions.tolist(),
            'y_test': y_test.tolist()
        }

    # Create the plotting object and create all the plots

    return stats


def save_stats(save_dir, stats, model_name):
    """
    asdfasdf

    @arguments:
        save_dir: <string>
        stats: <>
        model_name: <string>
    @returns:
        None
    """
    with open(save_dir + model_name + "_val_data" + '.json', 'w') as f:
        json.dump(stats, f)
    print("Stats saved")

#---MAIN-----------------+
def main():
    clear()
    # Model properties
    num_classes = 2

    # Load the model
    #model_name = "FirstTryNewData_classification_bCNN_1D_Fri_May_19_10:49:02.h5" 
    model_name = "TryingMonoZ_classification_bCNN_1D_Fri_May_19_11:55:47.h5"
    signature = "z"

    if "classification" in model_name:
        model_type = "cl"

    elif "regression" in model_name:
        model_type = "re"

    else:
        print("Could not identify whether the model type was classification or regression")
        exit(2)

    model_path = dirname + "/src/ML_1D/saved_models/" + model_name 
    model = load_model(model_path)
    
    # Load the dataset
    dataset = load_dataset(signature, model_type)
    print(dataset)
    
    # Evaluate and save the performance metrics
    stats = evaluate(model, dataset, num_classes, model_name)
    save_dir = dirname + "/src/ML_1D/testing_loadable_models/"
    save_stats(save_dir, stats, model_name)
    

if __name__ == "__main__":
    main()
