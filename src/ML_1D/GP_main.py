import sys
import time
import numpy as np
import os
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
import torch
import gpytorch

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#---LOCAL IMPORTS--------+
from ml_models.GP_cl import classification_GP
from ml_models.GP_re import regression_GP
from data_prep_1D.data_prep_2D_v2 import load_sets

#---GLOBALS--------------+
np.set_printoptions(threshold=np.inf)


def get_sets():
    """

    """
    cl, re = load_sets("jet", aug=False)
    input_shape = cl[0].shape[1], cl[0].shape[2] # Shape of X_train
    return cl, re, input_shape


def GP_classification(data_sets):
    """

    """
    if len(data_sets) != 4:
        print(f"Error in function <GP_classification>. Expected 6 datasets, got {len(data_sets)}")
        return

    X_train, y_train, X_test, y_test = data_sets

    model_name = "GP_classification_"
    timestamp = time.time()
    formatted_time = time.strftime("%a_%b_%d_%H:%M:%S", time.localtime(timestamp))
    model_name = model_name + formatted_time

    GaussProc_cl = classification_GP(X_train, y_train, X_test, y_test, model_name)
    GaussProc_cl.train()
    GaussProc_cl.evaluate(print_perf=False)


def main():
    cl_data_set, re_data_set, input_shape = get_sets()

    GP_classification(cl_data_set)


if __name__ == "__main__":
    main()
