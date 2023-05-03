#---IMPORTS----------+
import numpy as np
import os
import csv

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
                file_dict[model].append(folder_path + "/" + csv_file)
        file_dict[model].sort()
    
    # Create a dictionary with all the data and the mass as values, and the model name as key
    data_dict = {name: [[], [], [], []] for name in model_names} #[[MASS], [ETA], [MET], [PT]]
    
    for model in file_dict.keys():
        with open(file_dict[model][0]) as ETA_file, open(file_dict[model][1]) as MET_file, \
                open(file_dict[model][2]) as PT_file:

            reader_ETA = csv.reader(ETA_file)
            reader_MET = csv.reader(MET_file)
            reader_PT = csv.reader(PT_file)
            
            # Read all the rows in the 3 csv files
            for row in reader_ETA:
                try:
                    mass = np.float32(row[0])
                    values = [np.float32(val) for val in row[1:]]

                    data_dict[model][0].append(mass)
                    data_dict[model][1].append(values)

                except ValueError as e:
                    print(f"Error reading csv files: {e}")
            
            for row in reader_MET:
                try:
                    values = [np.float32(val) for val in row[1:]]
                    data_dict[model][2].append(values)

                except ValueError as e:
                    print(f"Error reading csv files: {e}")

            for row in reader_PT:
                try:
                    values = [np.float32(val) for val in row[1:]]
                    data_dict[model][3].append(values)

                except ValueError as e:
                    print(f"Error reading csv files: {e}")
    
    """
    print(data_dict["neutrino"][0], "\n")
    print(data_dict["neutrino"][1][1], "\n") 
    print(data_dict["neutrino"][2][1], "\n")
    print(data_dict["neutrino"][3][1], "\n")
    """
    return data_dict


def create_sets(data_dict):
    """
    @arguments:
        data_dict: <dictionary> Dictionary with all the data and masses
            as values, and the model name as key
    @returns:
        label_dict: <dicionary> Mapping of model name to encoding
        input_data: <numpy.array> Array of all the ETA, MET and PT data, stored in 3 channels.
        mass_data: <numpy.array> Array of all the masses corresponding to the input data.
        model_data: <numpy.array> Array of all the models corresponding to the input data.
    """
    # Define the data types for each channel, O for Object (numpy.array) 
    dtype = [('ETA', 'O'), ('MET', 'O'), ('PET', 'O')]
    
    # Create an empty structured array with one element
    # Should use preallocation
    input_data = np.empty(shape=(0,), dtype=dtype)
    mass_data = np.empty(shape=(0,)) 
    model_data = np.empty(shape=(0,))
    
    # Iterate over all the 'histograms'
    for model in data_dict.keys():
        for i in range(len(data_dict[model][1])):
            # Append all the data
            model_data = np.append(model_data, model)
            input_data = np.append(input_data, np.array([(np.array(data_dict[model][1][i]), \
                    np.array(data_dict[model][2][i]), np.array(data_dict[model][3][i]))], dtype=dtype))
            mass_data = np.append(mass_data, data_dict[model][0][i])
    
    # Use label encoding for the model names
    label_dict = create_label_encoding(model_data)
    model_data = [label_dict[model_data[i]] for i in range(len(model_data))]
    model_data = np.array(model_data)
    """ 
    print(mass_data, "\n")
    print(model_data, "\n")
    print(input_data[0], "\n")
    """
    return label_dict, input_data, mass_data, model_data


def create_label_encoding(class_label):
    """
    Hot encoding for the classification model.

    @arguments:
        class_label: <numpy.array> All the different models 
    @returns:
        target_dict: <dictionary> Mapping from model to encoding
    """
    target_dict = {k: v for v, k in enumerate(np.unique(class_label))}
    return target_dict
       

def shuffle_and_create_sets(X_data, labels, targets, random_seed = 13, print_shapes = False):
    """
    Shuffles all the images and labels/targets and creates three datasets with a ratio of 0.64:0.16:0.20.
    The training and testing sets are passed to the model in order to train it, while the validation set
    is an out of sample test, used for performance metrics.

    @arguments:
        X_data:    <numpy.ndarray> Array of all the input data stored in 3 channels.
        labels: <numpy.ndarray> Array of class labels for classification
        targets: <numpy.ndarray> Array of target values for regression
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

    # Check if all arrays have the same length as the first array
    same_length = np.all(np.array([len(X_data), len(labels), len(targets)]) == len(X_data))

    if not same_length: # Return empty sets if error
        print("Error in function <shuffle_and_create_sets>. Arrays are not the same size.")
        return [], [], [], [], [], []
    
    # Randomly shuffle the sets
    np.random.seed(random_seed)
    permutation = np.random.permutation(len(X_data))

    X_data_shuffled = X_data[permutation]
    labels_shuffled = labels[permutation]
    targets_shuffled = targets[permutation]
    
    # Create training, validation and testing sets using 0.64:0.16:0.20
    tot_len = len(X_data)
    first_split = int(tot_len * 0.64)
    second_split = int(first_split + tot_len * 0.16)
    
    # cl for classification, re for regression
    X_train, X_val, X_test = X_data_shuffled[0:first_split], X_data_shuffled[first_split:second_split], X_data_shuffled[second_split:]
    y_train_cl, y_val_cl, y_test_cl = labels_shuffled[0:first_split], labels_shuffled[first_split:second_split], labels_shuffled[second_split:]
    y_train_re, y_val_re, y_test_re = targets_shuffled[0:first_split], targets_shuffled[first_split:second_split], targets_shuffled[second_split:]
    
    # Print the shapes of the resulting arrays
    if print_shapes:
        print("Training set shapes: X_train={}, y_train_cl={}, y_train_re={}".format(X_train.shape, y_train_cl.shape, y_train_re.shape))
        print("Validation set shapes: X_val={}, y_val_cl={}, y_val_re={}".format(X_val.shape, y_val_cl.shape, y_val_re.shape))
        print("Testing set shapes: X_test={}, y_test_cl={}, y_test_re={}".format(X_test.shape, y_test_cl.shape, y_test_re.shape))
    
    return X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re

