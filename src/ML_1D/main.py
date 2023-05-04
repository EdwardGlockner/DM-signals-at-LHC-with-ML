# main.py

#---IMPORTS--------------+
import sys
import time
import numpy as np
import os
import getopt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore tensorflow warning

#---Argument Parsing-----+
# Usage: python main.py -r <run_mode> -m <model_prefix>

def usage():
    print("\nUsage:")
    print("main_training.py -r run_mode | -m model_prefix | -h help\n")
    print("-r \t Options: 'train', 'trainval'")
    print("Specifies if the program should be run in training mode, or training and validation mode.")
    print("run_mode is by default set to 't'.\n")
    print("-m \t Options: any string")
    print("Gives a prefix to the model name. The files will be named: <model_prefix>_Classification_bCNN_1D_<timestamp>.")
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
dirname = dirname.replace("src/ML_1D","")
sys.path.insert(1, os.path.join(dirname, "src/data_preparation/data_prep_1D"))
sys.path.insert(1, os.path.join(dirname, "src/ML_1D/ml_models"))

#---LOCAL IMPORTS--------+
from classification_bCNN import classification_bCNN
from regression_CNN import regression_CNN 
from data_prep_1D import create_sets_from_csv, shuffle_and_create_sets

#---GLOBALS--------------+
try:
    if sys.platform in ["darwin", "linux", "linux2"]: #macOS
        clear = lambda: os.system("clear")

    elif sys.platform in ["win32", "win64"]: #windows
        clear = lambda: os.system("cls")
    
    else:
        clear = lambda: None

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"


#---FUNCTIONS------------+
def preprocess():
    """

    """
    pass


def get_sets(folder_csv_path):
    """

    """
    X_data, targets, labels = create_sets_from_csv(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl,\
            y_test_re, X_val, y_val_cl, y_val_re =\
            shuffle_and_create_sets(X_data, labels, targets, print_shapes=False)

    input_shape = (X_train.shape[1], X_train.shape[2])

    return [X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re], input_shape

def train_classification(data_sets, input_shape, num_classes, model_prefix):
    """

    """
    if len(data_sets) != 4:
        print(f"Error in function <train_classification>. Expected 4 datasets, got {len(data_sets)}")

        
    X_train, y_train, X_test, y_test = data_sets 

    model_name = "classification_bCNN_1D_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    model = classification_bCNN(X_train, y_train, X_test, y_test, input_shape, num_classes, model_name) 
    model.compile()
    model.train()

    return model


def train_regression(data_sets, input_shape, model_prefix):
    """

    """
    if len(data_sets) != 4:
        print(f"Error in function <train_regression>. Expected 4 datasts, got {len(data_sets)}")

    X_train, y_train, X_test, y_test = data_sets

    model_name = "regression_CNN_1D"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    model = regression_CNN(X_train, y_train, X_test, y_test, input_shape, model_name)
    model.compile()
    model.train()

    return model


def val_classification(data_sets, input_shape, num_classes, model):
    """

    """
    if len(data_sets) != 2:
        print(f"Error in function <val_classification>. Expected 2 datasets, got {len(data_sets)}")
    
    X_val, y_val = data_sets
    model.evaluate_model(X_val, y_val)


def val_regression(data_sets, input_shape, model):
    """

    """
    if len(data_sets) != 2:
        print(f"Error in function <val_regression>. Expected 2 datasets, got {len(data_sets)}")

    X_val, y_val = data_sets

#---MAIN-----------------+
def main(run_mode, model_prefix):
    clear()
    folder_csv_path = dirname + "src/data_preparation/data_prep_1D/raw_data_all.csv"
    data_sets, input_shape = get_sets(folder_csv_path) 
    X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, X_val, y_val_cl, y_val_re = data_sets
    cl_data_set = [X_train, y_train_cl, X_test, y_test_cl]
    re_data_set = [X_train, y_train_re, X_test, y_test_re]

    cl_data_set_val = [X_val, y_val_cl]
    re_data_set_val = [X_val, y_val_re]

    num_classes = len(np.unique(y_train_cl))

    cl_model = train_classification(cl_data_set, input_shape, num_classes, model_prefix)
    re_model = train_regression(re_data_set, input_shape, model_prefix) 

    if run_mode == "trainval":
        val_classification(cl_data_set_val, input_shape, num_classes, cl_model)
        val_regression(re_data_set_val, input_shape, re_model)


#---RUN CODE-------------+
if __name__ == "__main__":
    run_mode, model_prefix = arg_parse(sys.argv[1:])
    main(run_mode, model_prefix)

