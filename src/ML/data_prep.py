#---IMPORTS----------+
import numpy as np
import os
from pathlib import Path
import matplotlib.image as mpimg
import csv
from matplotlib import pyplot as plt
import subprocess
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

#---FUNCTIONS--------+
def plot_img(path):
    """
    Plot and show an image given a path

    @arguments:
        path: <Path> Path to the location of the image
    @returns:
        None
    """
    plt.figure(figsize=(7,7))
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()


def create_label_encoding(class_label):
    """
    asdfasdf

    @arguments:
        class_label: <> 
    @returns:
        target_dict: <>
    """
    target_dict = {k: v for v, k in enumerate(np.unique(class_label))}
    return target_dict


def average_img(folder_path, show=False, save_file="Average.png"):
    """
    Takes in a path to a folder, and averages the pixelvalues of the images
    in the folder. The output image is saved and returned.

    This functions requires all images to be the same size

    @params:
        folder_path: <string> Full path to the folder where the images are
        show: <bool> Visualize image or note
        save_file <string> Name of the saved file

    @returns:
        avg_img: <> asdf 
    """
    # Read all files
    files = os.listdir(folder_path)
    imgs = [folder_path + file for file in files if file[-8:] in ["ETA1.png", "ETA1.PNG", "ETA1.jpg", "ETA1.JPG"]]
    imgs = [np.array(Image.open(img)) for img in imgs]
    arrs = [np.array(img) for img in imgs]
    avg_arr = np.mean(arrs, axis=0).astype(float)
    
    avg_img = Image.fromarray(avg_arr)
    avg_img = avg_img.convert("L") # Convert to grayscale
    # For RGD: ...convert("RGB")
    avg_img.save(save_file)
    
    if show:
        avg_img.show()
    
    return avg_img 


def regression_create_datasets(folder_img_path, output_path,  img_height, img_width):
    """
    asdfasdf

    @arguments:
        folder_path: <Path> The full path to the folder of data
        labels_path: <Path> The full path to the csv file with the labels
        img_height: <int> Integer value of the height of the images
        img_width: <int> Integer value of the width of the iamges
    @returns:
        None
    """
    img_data_arr = []
    target_vals  = []

    for file_name in folder_img_path:
        img_path = os.path.join(folder_img_path, file_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        img = img.astype("float32")
        img /= 255.0 # Normalizing
        img_data_arr.append(img)

    with open(output_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            target_vals.append(row)

    if len(img_data_arr) != len(target_vals):
        print("Error in function <regression_create_datasets>. Arrays are not the same size.")
        return None

    img_data_arr = np.array(img_data_arr)
    target_vals = np.array(target_vals)

    return img_data_arr, target_vals 
    

def classification_create_datasets(folder_path, img_height, img_width, label_encode = True):
    """
    asdfasdf
    Assumes all images are the same height and width

    @arguments:
        folder_path: <Path> The full path to the folder of data 
        img_height: <int> Integer value of the height of the images
        img_width: <int> Integer value of the width of the iamges
        label_encode: <bool> Whether to encode the categorical labels are not  
    @returns:
        img_data_arr: <numpy.ndarray> Array containing all the images represented by numpy.ndarrays
        class_label: <numpy.ndarray> Array of all the labels corresponding to img_data_arr (encoded by default)
    """
    img_data_arr = []
    class_label = []

    for file_name in folder_path:
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        img = img.astype("float32")
        img /= 255.0 # Normalizing
        img_data_arr.append(img)
        class_label.append("model")

    if label_encode:
        label_dict = create_label_encoding(class_label)
        class_label = [label_dict[class_label[i]] for i in range(len(class_label))]

    if len(img_data_arr) != len(class_label):
        print("Error in function <classification_create_datasets>. Arrays are not the same size.")
        return None

    img_data_arr = np.array(img_data_arr)
    class_label = np.array(class_label)

    return img_data_arr, class_label


def shuffle_and_create_sets(img_data_arr, label_or_target, random_seed = 13, print_shapes = False):
    """
    asdfasdf

    @arguments:
        img_data_arr: <numpy.ndarray> Array of all the images represented as numpy.ndarrays
        label_or_target: <numpy.ndarray> Array of either class labels for classification or target values for regression
        random_seed: <int> Random seed for the shuffling
        print_shapes: <bool> Whether to print the shapes of the sets or not
    @returns:
        X_train: <>
        y_train: <>
        X_test: <>
        y_test: <>
        X_val: <>
        y_val: <>
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    if len(img_data_arr) != len(label_or_target):
        print("Error in function <shuffle_and_create_sets>. Arrays are not the same size.")
        return X_train, y_train, X_test, y_test 
    
    # Randomly shuffle the sets
    np.random.seed(random_seed)
    permutation = np.random.permutation(len(img_data_arr))

    img_data_arr_shuffled = img_data_arr[permutation]
    label_or_target_shuffled = label_or_target[permutation]
    
    # Create training, validation and testing sets using 0.64:0.16:0.20
    X_train, X_temp, y_train, y_temp = train_test_split(img_data_arr_shuffled, label_or_target_shuffled, test_size=0.36)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.44/0.56)
   
    # Print the shapes of the resulting arrays
    if print_shapes:
        print("Training set shapes: X_train={}, y_train={}".format(X_train.shape, y_train.shape))
        print("Validation set shapes: X_val={}, y_val={}".format(X_val.shape, y_val.shape))
        print("Testing set shapes: X_test={}, y_test={}".format(X_test.shape, y_test.shape))
   
    return X_train, y_train, X_test, y_test, X_val, y_val 


def combine_imgs(script_path, folder_path, folder_dest=""):
    """
    Runs a bash script to combine multiple images into one bigger image.

    @arguments:
        script_path: <string> Full path to the bash script that will be run
        folder_path: <string> Path to where the images are stored 
        folder_dest: <string> To which folder the combined images should be saved
    @returns:
        None
    """
    if folder_dest == "":
        folder_dest = folder_path
    
    try:
        subprocess.run(script_path, shell=True)

    except subprocess.CalledProcessError as e:
        print ( "Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr.decode("utf-8") )
        raise


def lower_res(script_path, folder_path, folder_dest=""):
    """
    Runs a bash script to lower the resolution of images in a folder.

    @arguments:
        script_path: <string> Full path to the bash script that will be run
        folder_path: <string> Path to where the images are stored
        folder_dest: <string> To which folder the combined images should be saved
    @returns:
        None
    """
    if folder_dest == "":
        folder_dest = folder_path

    try:
        subprocess.run(script_path, shell=True)

    except subprocess.CalledProcessError as e:
        print ( "Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr.decode("utf-8") )
        raise

