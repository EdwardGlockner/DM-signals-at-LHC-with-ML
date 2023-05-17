# main.py

#---IMPORTS--------------+
import sys
import time
import numpy as np
import os
import argparse
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
import torch
import gpytorch

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
        print("Gives a prefix to the model name. The files will be named: <model_prefix>_Classification_bCNN_1D_<timestamp>.")
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
from ml_models.classification_bCNN import classification_bCNN
from ml_models.regression_CNN import regression_CNN
from ml_models.GP_cl import classification_GP
from ml_models.GP_re import regression_GP
from data_prep_1D.data_prep_1D import create_sets_from_csv, \
        shuffle_and_create_sets
from data_prep_1D.data_prep_2D_v2 import load_sets

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
def get_sets(*folder_csv_path):
    """

    """
    """
    X_data, targets, labels = create_sets_from_csv(*folder_csv_path) 
    #count_zero = np.count_nonzero(labels == 0)
    #count_one = np.count_nonzero(labels == 1)
    #print(f"Number of zeros: {count_zero}")
    #print(f"Number of ones: {count_one}")

    cl, re = shuffle_and_create_sets(X_data, labels, targets, print_shapes=False)
    X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl = cl
    X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, X_test_re_cat, \
            y_test_re, X_val_re_hist, X_val_re_cat, y_val_re = re
    
    input_shape = (X_train_cl.shape[1], X_train_cl.shape[2])
    #print(np.count_nonzero(y_train_cl == 0))
    #print(np.count_nonzero(y_train_cl == 1))
    #print(np.count_nonzero(y_test_cl == 0))
    #print(np.count_nonzero(y_test_cl == 1))
    #print(np.count_nonzero(y_val_cl == 0))
    #print(np.count_nonzero(y_val_cl == 1))
    return [X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl], \
            [X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, X_test_re_cat, \
            y_test_re, X_val_re_hist, X_val_re_cat, y_val_re], input_shape
    """
    cl, re = load_sets()
    input_shape = cl[0].shape[1], cl[0].shape[2] # Shape of X_train
    return cl, re, input_shape

def train_classification(data_sets, input_shape, num_classes, model_prefix):
    """

    """
    if len(data_sets) != 4:
        print(f"Error in function <train_classification>. Expected 4 datasets, got {len(data_sets)}")
        return
        
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
    if len(data_sets) != 6:
        print(f"Error in function <train_regression>. Expected 4 datasts, got {len(data_sets)}")
        return

    X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, y_test = data_sets

    model_name = "regression_CNN_1D_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    model = regression_CNN(X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, y_test, input_shape, model_name)
    model.compile()
    model.train()

    return model


def val_classification(data_sets, input_shape, num_classes, model, img_save_path):
    """

    """
    if len(data_sets) != 2:
        print(f"Error in function <val_classification>. Expected 2 datasets, got {len(data_sets)}")
        return
    
    X_val, y_val = data_sets
    model.evaluate_model(X_val, y_val, img_save_path)


def val_regression(data_sets, input_shape, model, img_save_path):
    """

    """
    if len(data_sets) != 3:
        print(f"Error in function <val_regression>. Expected 3 datasets, got {len(data_sets)}")
        return

    X_val_hist, X_val_cat, y_val = data_sets
    model.evaluate_model(X_val_hist, X_val_cat, y_val, img_save_path)


def GP_classification(data_sets):
    """

    """
    if len(data_sets) != 6:
        print(f"Error in function <GP_classification>. Expected 6 datasets, got {len(data_sets)}")
        return

    X_train, y_train, X_test, y_test, X_val, y_val = data_sets

    model_name = "GP_classification_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time

    GaussProc_cl = classification_GP(X_train, y_train, X_test, y_test, X_val, \
            y_val, model_name)
    GaussProc_cl.train()
    GaussProc_cl.evaluate(print_perf=False)


def GP_regression(data_sets):
    """

    """
    if len(data_sets) != 9:
        print(f"Error in function <GP_regression>. Expected 9 datasets, got {len(data_sets)}")

    X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, y_test, \
            X_val_hist, X_val_cat, y_val = data_sets

    model_name = "GP_regression_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_prefix + "_" + model_name + formatted_time
    
    likelihood = GaussianLikelihood()

    X_train_hist = torch.from_numpy(X_train_hist).float()
    X_train_cat = torch.from_numpy(X_train_cat).float()
    y_train = torch.from_numpy(y_train).float()

    X_test_hist = torch.from_numpy(X_test_hist).float()
    X_test_val = torch.from_numpy(X_test_cat).float()
    y_test = torch.from_numpy(y_test).float()

    X_val_hist = torch.from_numpy(X_val_hist).float()
    X_val_cat = torch.from_numpy(X_val_cat).float()
    y_val = torch.from_numpy(y_val).float
    GaussProc_re = regression_GP(X_train_hist, X_train_cat, y_train, X_test_hist, \
            X_test_cat, y_test, X_val_hist, X_val_cat, y_val, likelihood, model_name)

    GaussProc_re.train_model(likelihood,num_iterations=100, lr=0.1)

    GaussProc_re.eval()
    with torch.no_grad():
        predictions = GaussProc_re(X_test_hist)

    mse = torch.mean((predictions - y_test) ** 2)
    rmse = torch.sqrt(mse)

    print("Mean Squared Error (MSE):", mse.item())
    print("Root Mean Squared Error (RMSE):", rmse.item())


#---MAIN-----------------+
def main(run_mode, model_type, model_prefix):
    clear()
    # Create all the datasets
    sneutrino_jet_path = dirname + "/Storage_data/MSSM_sneutrino_jet/norm_amp_array/raw_data_all.csv"
    sneutrino_jet_path2 = dirname + "/Storage_data/MSSM_sneutrino_jet/data_all.csv"
    neutralino_jet_path = dirname + "/Storage_data/MSSM_neutralino_jet/norm_amp_array/raw_data_all.csv"
    neutralino_jet_path2 = dirname + "/Storage_data/MSSM_neutralino_jet/data_all.csv"
    sneutrino_z_path = dirname + "/Storage_data/MSSM_neutralino_z/norm_amp_array/raw_data_all.csv"
    neutralino_z_path = dirname + "/Storage_data/MSSM_neutralino_z/norm_amp_array/raw_data_all.csv"

    data_set_cl, data_set_re, input_shape = get_sets(sneutrino_jet_path, \
            sneutrino_jet_path2, neutralino_jet_path, neutralino_jet_path2) 

    X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl = data_set_cl
    X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, X_test_re_cat, \
            y_test_re, X_val_re_hist, X_val_re_cat, y_val_re = data_set_re
    
    cl_data_set = [X_train_cl, y_train_cl, X_test_cl, y_test_cl]
    re_data_set = [X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, \
            X_test_re_cat,  y_test_re]

    cl_data_set_val = [X_val_cl, y_val_cl]
    re_data_set_val = [X_val_re_hist, X_val_re_cat,  y_val_re]

    num_classes = len(np.unique(y_train_cl))
    
    # Train the models
    if model_type == "cl":
        cl_model = train_classification(cl_data_set, input_shape, num_classes, model_prefix)
        # Run Gaussian Process
        GP_classification([X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl])

    elif model_type == "re":
        re_model = train_regression(re_data_set, input_shape, model_prefix) 
        # Run Gaussian Process
        #GP_regression([X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, \
    #        X_test_re_cat, y_test_re, X_val_re_hist, X_val_re_cat, y_val_re])

    elif model_type == "clre":
        cl_model = train_classification(cl_data_set, input_shape, num_classes, model_prefix)
        re_model = train_regression(re_data_set, input_shape, model_prefix) 

        # Run Gaussian Process
        GP_classification([X_train_cl, y_train_cl, X_test_cl, y_test_cl, X_val_cl, y_val_cl])
        #GP_regression([X_train_re_hist, X_train_re_cat, y_train_re, X_test_re_hist, \
        #        X_test_re_cat, y_test_re, X_val_re_hist, X_val_re_cat, y_val_re])


    else:
        print("Not a valid model_type. See python main.py -h for help.")
        return

    # Validate the models

    if run_mode == "trainval":
        img_save_path = dirname + "src/ML_1D/plots"
        if model_type == "cl":
            val_classification(cl_data_set_val, input_shape, num_classes, cl_model, img_save_path)

        elif model_type == "re":
            val_regression(re_data_set_val, input_shape, re_model, img_save_path)

        elif model_type == "clre":
            val_classification(cl_data_set_val, input_shape, num_classes, cl_model, img_save_path)
            val_regression(re_data_set_val, input_shape, re_model, img_save_path)
    


#---RUN CODE-------------+
if __name__ == "__main__":
    run_mode, model_type, model_prefix = arg_parse()
    main(run_mode, model_type, model_prefix)

