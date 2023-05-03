#---IMPORTS----------+
import numpy as np
import os
import sys
from pathlib import Path
import csv
from sklearn.model_selection import train_test_split

#---FUNCTIONS--------+

def get_models(folder_path):
    """
    Get the name of all the models.
    In the directory files are named for example {model_name}_ETA.csv
    This function returns all the unique model_names.

    @arguments:
        folder_path: <string> Full path to the folder containing the csv files.
    @returns:
        model_names: <list> List of all the model names.
    """
    # Read all the csv files in the directory
    file_names = os.listdir(folder_path)
    files = [file for file in file_names if file[-4:] == ".csv"]
    model_names = []

    # Find all the unique models
    for file_name in files:
        name = file_name.split("_")[0]
        if name not in model_names:
            model_names.append(name)

    return model_names


def read_csvs_to_data_set(folder_path, model_names):
    """
    Reads all the csv:s to create the 1D arrays with 3 channels.

    @arguments:
        folder_path: <string> Full path to the folder containing the csv files
        model_names: <list> Containing all the names of the different models.
    @returns:
        data_dict: <>
    """
    
    # Read all the csv files in the directory
    file_names = os.listdir(folder_path)
    files = [file for file in file_names if file[-4:] == ".csv"]
    
    # Create a dict with the model name as the key, and the three csv files as values in a list
    file_dict = {name: [] for name in model_names}

    for model in file_dict.keys():
        for csv_file in files:
            if csv_file.split("_")[0] == model:
                file_dict[model].append(folder_path + csv_file)

def create_label_encoding(class_label):
    """
    Hot encoding for the classification model.
    @arguments:
        class_label: <> 
    @returns:
        target_dict: <>
    """
    target_dict = {k: v for v, k in enumerate(np.unique(class_label))}
    return target_dict
       

def regression_create_targets(targets_path):
    """
    Reads all the data used for the regression model from a csv file. 

    @arguments:
        targets_path: <string> The full path to the csv file with the labels
    @returns:
        target_vals: <numpy.ndarray> Array containing all the output float values 
    """
    target_vals  = []
    
    # Opens and read the csv file
    with open(targets_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            try:
                row = np.float32(row[0])
            except ValueError:
                print("Error in function {regression_create_datasets}. Could not convert string to np.float32.")

            target_vals.append(row)
    
    # Convert to numpy array
    target_vals = np.array(target_vals)

    return target_vals 


def classification_create_labels(folder_path, label_encode = True):
    """
    Creates all the labels used for the classification model. Hot encoding is available.

    @arguments:
        folder_path:  <string> The full path to the folder of data 
        label_encode: <bool> Whether to encode the categorical labels are not  
    @returns:
        class_label: <numpy.ndarray> Array of all the labels corresponding to img_data_arr (encoded by default)
    """
    class_label = []
    
    # Parse all the model names
    for file_name in folder_path:
        class_label.append("model")
    
    # Use hot encoding
    if label_encode:
        label_dict = create_label_encoding(class_label)
        class_label = [label_dict[class_label[i]] for i in range(len(class_label))]
    
    class_label = np.array(class_label)

    return class_label


def shuffle_and_create_sets(img_data_arr, label_or_target, random_seed = 13, print_shapes = False):
    """
    Shuffles all the images and labels/targets and creates three datasets with a ratio of 0.64:0.16:0.20.
    The training and testing sets are passed to the model in order to train it, while the validation set
    is an out of sample test, used for performance metrics.

    @arguments:
        img_data_arr:    <numpy.ndarray> Array of all the images represented as numpy.ndarrays
        label_or_target: <numpy.ndarray> Array of either class labels for classification or target values for regression
        random_seed:     <int> Random seed for the shuffling
        print_shapes:    <bool> Whether to print the shapes of the sets or not
    @returns:
        X_train: <numpy.ndarray> Contains 64 % of the data
        y_train: <numpy.ndarray> Contains 64 % of the data
        X_test:  <numpy.ndarray> Contains 16 % of the data 
        y_test:  <numpy.ndarray> Contains 16 % of the data
        X_val:   <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
        y_val:   <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
    """
    if len(img_data_arr) != len(label_or_target): # Return empty sets if error
        print("Error in function <shuffle_and_create_sets>. Arrays are not the same size.")
        return [], [], [], [], [], []
    
    # Randomly shuffle the sets
    np.random.seed(random_seed)
    permutation = np.random.permutation(len(img_data_arr))

    img_data_arr_shuffled = img_data_arr[permutation]
    label_or_target_shuffled = label_or_target[permutation]
    
    # Create training, validation and testing sets using 0.64:0.16:0.20
    X_train, X_val_test, y_train, y_val_test = train_test_split(img_data_arr_shuffled, label_or_target_shuffled, test_size=0.36, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.44, random_state=42)
   
    # Print the shapes of the resulting arrays
    if print_shapes:
        print("Training set shapes: X_train={}, y_train={}".format(X_train.shape, y_train.shape))
        print("Validation set shapes: X_val={}, y_val={}".format(X_val.shape, y_val.shape))
        print("Testing set shapes: X_test={}, y_test={}".format(X_test.shape, y_test.shape))

    return X_train, y_train, X_test, y_test, X_val, y_val 


