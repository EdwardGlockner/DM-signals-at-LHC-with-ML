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


def average_imgs(folder_path, folder_dest, show=False):
    """
    Takes in a path to a folder, and averages the pixelvalues of the images
    in the folder. The output image is saved and returned.

    This functions requires all images to be the same size

    @params:
        folder_path: <string> Full path to the folder where the images are
        show: <bool> Visualize image or note
        folder_dest <string> Path to the folder where the files should be saved
    @returns:
        None
    """
    # Read all files
    files = os.listdir(folder_path)
    imgs = [folder_path + file for file in files if file[-4:] == ".png"]

    # Group all the images based on first index, and image type (last 7 characters)
    img_grouping = {}
    for i in range(len(imgs)):
        if ".png" in imgs[i]: # Check so it is a picture
            temp_img = imgs[i].split("/")[-1]
            temp_key = (temp_img[0], temp_img[-7:])

            if temp_key in img_grouping:
                img_grouping[temp_key].append(imgs[i])
            else:
                img_grouping[temp_key] = [imgs[i]]
    
    # Convert into list
    img_grouping = [v for k, v in img_grouping.items()]
    
    # Averaging all the images, and saving them
    for i in range(len(img_grouping)):
        imgs = [np.array(Image.open(img)) for img in img_grouping[i]]
        arrs = [np.array(img) for img in imgs]
        avg_arr = np.mean(arrs, axis=0).astype(float)
        
        avg_img = Image.fromarray(avg_arr)
        avg_img = avg_img.convert("L") # Convert to grayscale
        # For RGD: ...convert("RGB")
        
        new_file_name = img_grouping[i][0].split("/")[-1]
        new_file_name = new_file_name[0:2] + new_file_name[4:]

        avg_img.save(folder_dest + new_file_name)
        
        if show:
            avg_img.show()
        

def read_images(folder_img_path, img_height, img_width):
    """
    asdfasdf

    @arguments:
        folder_img_path: <Path> The full path to the folder of the image data
        img_height: <int> Integer value of the height of the images
        img_width: <int> Integer value of the width of the images
    @returns:
        img_data_arr: <numpy.ndarray> Array containing all the images represented by numpy.ndarrays
    """
    img_data_arr = []

    for file_name in folder_img_path:
        img_path = os.path.join(folder_img_path, file_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height, img_width), interpolation = cv2.INTER_AREA)
        img = np.array(img)
        img = img.astype("float32")
        img /= 255.0 # Normalization
        img_data_arr.append(img)

    img_data_arr = np.array(img_data_arr)

    return img_data_arr


def regression_create_targets(targets_path):
    """
    asdfasdf

    @arguments:
        targets_path: <Path> The full path to the csv file with the labels
    @returns:
        target_vals: <numpy.ndarray> Array containing all the output float values 
    """
    target_vals  = []

    with open(targets_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            try:
                row = np.float32(row[0])
            except ValueError:
                print("Error in function {regression_create_datasets}. Could not convert string to np.float32.")

            target_vals.append(row)

    target_vals = np.array(target_vals)

    return target_vals 


def classification_create_labels(folder_path, label_encode = True):
    """
    asdfasdf
    Assumes all images are the same height and width

    @arguments:
        folder_path: <Path> The full path to the folder of data 
        label_encode: <bool> Whether to encode the categorical labels are not  
    @returns:
        class_label: <numpy.ndarray> Array of all the labels corresponding to img_data_arr (encoded by default)
    """
    class_label = []

    for file_name in folder_path:
        class_label.append("model")

    if label_encode:
        label_dict = create_label_encoding(class_label)
        class_label = [label_dict[class_label[i]] for i in range(len(class_label))]
    
    class_label = np.array(class_label)

    return class_label


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
    
    script_path = f"{script_path} {folder_path} {folder_dest}"
    try:
        print(subprocess.run(script_path, check=True, capture_output=True, text=True, shell=True))

    except subprocess.CalledProcessError as e:
        print ( "Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr.decode("utf-8") )
        raise
    """
    import cv2

    img1 = cv2.imread("0_0_neutralino_ETA.png")
    img2 = cv2.imread("0_0_neutralino_MET.png")
    img3 = cv2.imread("0_0_neutralino_PT.png")

    img = cv2.hconcat([img1, img2, img3])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test.png", img)
    """
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

combine_imgs("/Users/MaxAn/Documents/VScode/Kandidatprojekt/DM-signals-at-LHC-with-ML/src/ML/data_preparation/combine_png.sh", \
              "/Users/MaxAn/Documents/VScode/Kandidatprojekt/DM-signals-at-LHC-with-ML/src/ML/raw_data/Test_images", \
              "/Users/MaxAn/Documents/VScode/Kandidatprojekt/DM-signals-at-LHC-with-ML/src/ML/raw_data/Combine")
