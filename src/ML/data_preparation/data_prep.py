#---IMPORTS----------+
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.image as mpimg
import csv
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

#---FUNCTIONS--------+
def plot_img(path):
    """
    Plot and show an image given a path

    @arguments:
        path: <string> Path to the location of the image
    @returns:
        None
    """
    plt.figure(figsize=(7,7))
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()


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
       

def read_images(folder_img_path):
    """
    Reads and converts all images in a given folder to a numpy array. Also returns the shape of the images.
    Assumes all images have the same size

    @arguments:
        folder_img_path: <Path> The full path to the folder of the image data
        img_height: <int> Integer value of the height of the images
        img_width: <int> Integer value of the width of the images
    @returns:
        img_data_arr: <numpy.ndarray> Array containing all the images represented by numpy.ndarrays
        img_shape: <tuple> Tuple of the size of the images (width, height, channels)
    """
    img_data_arr = []
    
    # Read all files in the directory
    file_names = os.listdir(folder_img_path)
    files = [folder_img_path + file for file in file_names if file[-4:] == ".png"]

    # Get size of the images
    temp_img = Image.open(files[0])
    img_width, img_height = temp_img.size

    for file_name in files:
        img = cv2.imread(file_name, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_height, img_width), interpolation = cv2.INTER_AREA)
        img = np.array(img)
        img = img.astype("float32")
        img /= 255.0 # Normalization
        img_data_arr.append(img)
    
    img_data_arr = np.array(img_data_arr)

    img_shape = (img_width, img_height, 1)

    return img_data_arr, img_shape


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
        folder_path: <string> The full path to the folder of data 
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
        img_data_arr: <numpy.ndarray> Array of all the images represented as numpy.ndarrays
        label_or_target: <numpy.ndarray> Array of either class labels for classification or target values for regression
        random_seed: <int> Random seed for the shuffling
        print_shapes: <bool> Whether to print the shapes of the sets or not
    @returns:
        X_train: <numpy.ndarray> Contains 64 % of the data
        y_train: <numpy.ndarray> Contains 64 % of the data
        X_test: <numpy.ndarray> Contains 16 % of the data 
        y_test: <numpy.ndarray> Contains 16 % of the data
        X_val: <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
        y_val: <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
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


