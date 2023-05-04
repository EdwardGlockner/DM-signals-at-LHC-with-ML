#main.py 

#---IMPORTS--------------+
import numpy as np
import sys
import os
import time
import getopt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore tensorflow warning

#---Argument Parsing-----+
# Usage: python main.py -r <run_mode> -m <model_prefix> 

def usage():
    print("\nUsage:")
    print("main.py -r run_mode | -m model_prefix | -h help\n")
    print("-r \t Options: 'train', 'trainval'")
    print("Specifies if the program should be run in training mode, or training and validation mode.")
    print("run_mode is by default set to 't'.\n")
    print("-m \t Options: any string")
    print("Gives a prefix to the model name. The files will be named: <model_prefix>_Classification_bCNN_2D_<timestamp>.")
    print("model_prefix is by default set to ''.\n")

def arg_parse(argv):
    """

    """
    run_mode = ""
    model_prefix = ""
    try:
        opts, args = getopt.getopt(argv, "r:m:", ["run_mode", "model_prefix"])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        if opt == "-r":
            run_mode = arg

        if opt == "-m":
            model_prefix = arg
    
    if not run_mode:
        run_mode = "train"

    if not model_prefix:
        model_prefix = ""

    return run_mode, model_prefix


#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("src/ML_2D","")
sys.path.insert(1, os.path.join(dirname, "src/data_preparation/data_prep_2D"))
sys.path.insert(1, os.path.join(dirname, "src/ML_2D/ml_models"))

#---LOCAL IMPORTS--------+
from classification_bCNN import classification_bCNN
from regression_CNN import regression_CNN 
from data_prep_2D import read_images, shuffle_and_create_sets, classification_create_labels, regression_create_targets      
from preprocess_2D import average_imgs, clear_img_directory, combine_imgs, lower_res   

#---GLOBALS--------------+
try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS
        clear = lambda : os.system("clear")

    elif sys.platform in ["win32", "win64"]: #windows
        clear = lambda : os.system("cls")
    
    else:
        clear = ""

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"


#---FUNCTIONS------------+
def preprocess(folder_img_path, folder_dest, clear_dir=True):
    """
    asdfasdf

    @arguments:
        folder_img_path: <string>
        folder_dest:     <string>
        clear_dir:       <bool>
    @returns:
        None
    """
    if clear_dir:
        clear_img_directory(folder_dest)

    average_imgs(folder_img_path, folder_dest, show=False)
    combine_imgs(folder_dest, folder_dest, True)
    lower_res(folder_dest)


def get_sets(folder_img_path, folder_target_path):
    """
    asdfasdf

    @arguments:
        None
    @returns:
        None
    """
    img_arr, img_shape = read_images(folder_img_path)
    target_vals = regression_create_targets(folder_target_path) 
    #class_labels = classification_create_labels(folder_img_path)

    
    if len(img_arr) != len(target_vals):
        print("Error in function <get_sets>. Arrays img_arr and target_vals are not the same size.")

    #if len(img_arr) != len(class_labels):
    #    print("Erorr in function <get_sets>. Arrays img_arr and class_labels are not the same size.")

    #X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl = \
    #     shuffle_and_create_sets(img_arr, class_labels)

    X_train_re, y_train_re, X_test_re, y_test_re, X_val_re, y_val_re = \
            shuffle_and_create_sets(img_arr, target_vals)

    return [X_train_re, y_train_re, X_test_re, y_test_re, X_val_re, y_val_re], img_shape 
    #return [X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl], \
    #        [X_train_re, y_train_re, X_test_re, y_test_re, X_val_re, y_val_re]
    

def train_classification(data_sets, input_shape, model_prefix):
    """
    asdfasdf

    @arguments:
        data_sets:   <list> List of the datasets. Order: X_train, y_train, X_test, y_test, X_val, y_val
        input_shape: <tuple> Tuple of the input shape (width, height, channels)
    @returns:
        None
    """
    if len(data_sets) != 6:
        print("Error in function <train_classification>. Expected data_sets to contain 6 sets.")
        return None
    
    X_train, y_train, X_test, y_test, X_val, y_val = data_sets 
    num_classes = len(np.unique(y_train))

    model_name = "classification_bCNN_2D_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    model = classification_bCNN(X_train, y_train, X_test, y_test, input_shape, num_classes, model_name)
    model.compile()
    model.train()


def train_regression(data_sets, input_shape, model_prefix):
    """
    asdfasdf

    @arguments:
        data_sets:   <list> List of the datasets. Order: X_train, y_train, X_test, y_test, X_val, y_val
        input_shape: <tuple> Tuple of the input shape (width, height, channels)
    @returns:
        None
    """
    if len(data_sets) != 6:
        print("Error in function <train_regression>. Expected data_sets to contain 6 sets.")
        return None

    X_train, y_train, X_test, y_test, X_val, y_val = data_sets 

    model_name = "regression_CNN_1D_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    model = regression_CNN(X_train, y_train, X_test, y_test, input_shape, model_name)
    model.compile()
    model.train()


#---MAIN-----------------+
def main(run_mode, model_prefix):
    # Create all the paths to the directories
    folder_img_path = dirname + "src/raw_data/images/"
    folder_dest_path = dirname + "src/processed_data/images/"
    folder_target_path = dirname + "src/raw_data/target_values/data_MSSM.csv" 

    # Preprocessing and creating datasets
    preprocess(folder_img_path, folder_dest_path) 

    re, image_shape = get_sets(folder_dest_path, folder_target_path) 
    #eventually: re, cl, image_shape = get_sets(....
    # Train the classification model

    # Train the regression model
    train_regression(re, image_shape, model_prefix) 

    """
    cl, re =  get_sets(folder_img_path, folder_target_path, img_height, img_width)
    X_val_cl, y_val_cl = cl[4], cl[5]
    X_val_re, y_val_re = re[4], re[5]

    train_classification(cl[0], cl[1], cl[2], cl[3])
    train_regression(re[0], re[1], re[2], re[3])
    """


#---RUN CODE-------------+
if __name__ == "__main__":
    run_mode, model_prefix = arg_parse(sys.argv[1:])
    main(run_mode, model_prefix)

