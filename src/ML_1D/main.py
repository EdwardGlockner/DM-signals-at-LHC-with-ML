# main.py

#---IMPORTS--------------+
import sys
import time
import numpy as np
import os
import argparse
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore tensorflow warning

#---Argument Parsing-----+
def arg_parse():
    def usage():
        print("\n")
        print(" /$$$$$$  /$$   /$$ /$$   /$$")
        print(" /$$__  $$| $$$ | $$| $$$ | $$")
        print("| $$  \__/| $$$$| $$| $$$$| $$")
        print("| $$      | $$ $$ $$| $$ $$ $$")
        print("| $$      | $$  $$$$| $$  $$$$")
        print("| $$    $$| $$\  $$$| $$\  $$$")
        print("|  $$$$$$/| $$ \  $$| $$ \  $$")
        print(" \______/ |__/  \__/|__/  \__/")

        print("\n------------------Usage-----------------")
        print("main_training.py --run run_mode | --type model_type | --name model_prefix | -h help\n")

        print("--run \t Options: 'train', 'trainval'")
        print("Specifies if the program should be run in training mode, or training and validation mode.")
        print("run_mode is by default set to 'train'.\n")

        print("--type \t Options: 'cl', 're', clre'")
        print("Specifies if the program should run the classification model, or the regression model")
        print("model_type is by default set to 'cl'.\n")

        print("--name \t Options: any string")
        print("Gives a prefix to the model name. The files will be named: <model_prefix>_Classification_CNN_1D_<timestamp>.")
        print("model_prefix is by default set to 'test'.\n")


    parser = argparse.ArgumentParser(description="Machine Learning DM", add_help=False)
    parser.add_argument('--run', type=str, default='train', help='Run mode')
    parser.add_argument('--type', type=str, default='cl', help='Model type')
    parser.add_argument('--name', type=str, default='test', help='Model name')
    parser.add_argument('-h', '--help', action='store_true', help='Show help message')
    args = parser.parse_args()

    if args.help:
        usage()
        exit()

    return args.run, args.type, args.name


#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#---LOCAL IMPORTS--------+
from ml_models.classification_CNN import classification_CNN
from ml_models.regression_CNN import regression_CNN
from data_prep_1D.data_prep import load_sets

#---GLOBALS--------------+
np.set_printoptions(threshold=np.inf)

#---FUNCTIONS------------+
def get_sets(signature):
    """

    """
    cl, re = load_sets(signature, aug=False)
    input_shape = cl[0].shape[1], cl[0].shape[2] # Shape of X_train
    return cl, re, input_shape


def train_classification(data_sets, input_shape, num_classes, signature, learning_rate, model_name):
    """

    """
    if len(data_sets) != 4:
        print(f"Error in function <train_classification>. Expected 4 datasets, got {len(data_sets)}")
        return
        
    X_train, y_train, X_test, y_test = data_sets 

    model = classification_CNN(X_train, y_train, X_test, y_test, input_shape, \
            num_classes, signature, learning_rate, model_name) 
    model.compile()
    model.train()

    return model


def train_regression(data_sets, input_shape, signature, learning_rate, model_name):
    """

    """
    if len(data_sets) != 6:
        print(f"Error in function <train_regression>. Expected 4 datasts, got {len(data_sets)}")
        return

    X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, y_test = data_sets

    model = regression_CNN(X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, \
            y_test, input_shape, signature, learning_rate, model_name)
    model.compile()
    model.train()


def train(model_name, signature):
    """
    asdfasdf

    @arguments:
        signature: <>
    @returns:
        None
    """
    cl_data_set, re_data_set, input_shape = get_sets(signature)
    num_classes = len(np.unique(cl_data_set[1]))
    
    if signature == "z":
        learning_rate_cl = 0.000027
        learning_rate_re = 0.0045
    else: # jet
        learning_rate_cl = 0.000027
        learning_rate_re = 0.0798989898989899
    # Train the models
    if model_type == "cl":
        train_classification(cl_data_set, input_shape, num_classes, signature, \
                             learning_rate_cl, model_name + "_classification")

    elif model_type == "re":
        train_regression(re_data_set, input_shape, signature, learning_rate_re, \
                model_name + "_regression") 

    elif model_type == "clre":
        train_classification(cl_data_set, input_shape, num_classes, signature, \
                learning_rate_cl, model_name + "_classification")
        train_regression(re_data_set, input_shape, signature, learning_rate_re, \
                model_name + "_regression") 

    else:
        print("Not a valid model_type. See python main.py -h for help.")
        return

#---MAIN-----------------+
def main(run_mode, model_type, model_prefix):
    # Create model names
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name_z = model_prefix + "_z_" + formatted_time
    model_name_jet = model_prefix + "_jet_" + formatted_time
    
    # Train the models
    if model_type == "re":
        train(model_name_z, "z")

    if model_type == "cl":
        train(model_name_jet, "jet")
    
    # Validate the models
    if run_mode == "trainval":
        python_executable = "/usr/local/bin/python3.10" # find a better way to do this

        command1 = [python_executable, 'test.py', '--model_name', model_name_z + "_regression"]
        command2 = [python_executable, 'test.py', '--model_name', model_name_z + "_classification"]
        command3 = [python_executable, 'test.py', '--model_name', model_name_jet + "_classification"]
        command4 = [python_executable, 'test.py', '--model_name', model_name_jet + "_classification"]

        if model_type == "cl":
            print("Running mono_z classification model on testing data")
            subprocess.run(command2)
            print("Running mono_jet classification model on testing data")
            subprocess.run(command4)

        elif model_type == "re":
            print("Running mono_z regression model on testing data")
            subprocess.run(command1)
            print("Running mono_jet regression model on testing data")
            subprocess.run(command3)

        elif model_type == "clre":
            print("Running mono_z classification model on testing data")
            subprocess.run(command2)
            print("Running mono_jet classification model on testing data")
            subprocess.run(command4)
            print("Running mono_z regression model on testing data")
            subprocess.run(command1)
            print("Running mono_jet regression model on testing data")
            subprocess.run(command3)


#---RUN CODE-------------+
if __name__ == "__main__":
    run_mode, model_type, model_prefix = arg_parse()
    main(run_mode, model_type, model_prefix)

