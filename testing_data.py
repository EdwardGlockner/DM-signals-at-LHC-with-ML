#---IMPORTS----------+
import numpy as np
import pandas as pd
import os
import sys

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("","")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#---FUNCTIONS--------+

def create_sets_from_csv(file_path):
    """
    asdfasdf

    @arguments:
        folder_path: <string> Full path to the csv file
    @returns:

    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None)

    # Create a new dataframw with columns, [mass, model, eta, pt, met]
    # model values = {neutralino_jet, 0; neutrino_jet, 1}
    df =pd.DataFrame({
        'mass': df.iloc[:, 0],
        'model': df.iloc[:, 1],
        'eta': df.iloc[:, 2:47].values.tolist(),
        'pt': df.iloc[:, 47:92].values.tolist(),
        'tet': df.iloc[:, 92:].values.tolist()
    })
    print(f"Before: {df.shape[0]}")
    # Remove rows that have NaN values
    df['eta'] = df['pt'].apply(lambda x: [val for val in x if not np.isnan(val)])
    df['pt'] = df['pt'].apply(lambda x: [val for val in x if not np.isnan(val)])
    df['tet'] = df['tet'].apply(lambda x: [val for val in x if not np.isnan(val)])

    df = df[df['eta'].str.len() > 0]
    df = df[df['pt'].str.len() > 0]
    df = df[df['tet'].str.len() > 0]

    print(f"After: {df.shape[0]}") 
    # Create separate arrays
    masses = df["mass"].values
    models = df["model"].values
    eta_vals = df["eta"].values
    pt_vals = df["pt"].values
    tet_vals = df["tet"].values
    
    # Convert the elements into numpy.ndarrays
    eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
    pt_vals = np.array([np.array(sublist) for sublist in pt_vals])
    tet_vals = np.array([np.array(sublist) for sublist in tet_vals])
    # Concatenate into three channels
    input_data  = np.concatenate([eta_vals[..., np.newaxis], pt_vals[..., np.newaxis], tet_vals[..., np.newaxis]], axis=2)

    return input_data, masses, models


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



###################################################################################################
def get_sets(folder_csv_path):
    """

    """
    X_data, targets, labels = create_sets_from_csv(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl,\
            y_test_re, X_val, y_val_cl, y_val_re =\
            shuffle_and_create_sets(X_data, labels, targets, print_shapes=False)

    input_shape = (X_train.shape[1], X_train.shape[2])

    return [X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re], input_shape


def main():
    folder_csv_path = dirname + "/Storage_data/MSSM_sneutrino_jet/norm_amp_array/raw_data_all.csv"
    #folder_csv_path = dirname + "/Storage_data/MSSM_neutralino_jet/norm_amp_array/raw_data_all.csv"
    data_sets, input_shape = get_sets(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re = data_sets
    print(y_train_cl)
if __name__ == "__main__":
    main()

