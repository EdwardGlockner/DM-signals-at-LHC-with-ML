#---IMPORTS--------------+
import keras
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import time

#---LOCAL IMPORTS--------+
from data_prep import *
from CNN import *

#---FIXING PATH----------+
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("CNN_mnist/src","")

#---GLOBALS--------------+
try:
    if sys.platform == "darwin": #macOS
        clear = lambda : os.system("clear")

    elif sys.platform == "win32" or sys.platform == "win64": #windows
        clear = lambda : os.system("cls")

    elif sys.platform == "linux" or sys.platform == "linux2":
        clear = lambda : os.system("clear")

except OSError as e:
    print("Error identifying operating systems")

bar = "+---------------------+"

#---MAIN-----------------+
def main():
    clear()
    check = True
    while check:
        print(bar)
        print("Choose dataset")
        print("1. MNIST")
        print("2. MNIST Fashion")
        choice_set = input("\nChoose dataset: ")
        clear()
        print(choice_set)
        if choice_set == "1":
            dataset = "MNIST"
            check = False

        if choice_set == "2":
            dataset = "MNIST_Fashion"
            check = False

    check = True
    while check:
        clear()
        print(bar)
        print("Model settings")
        print("1. Load model")
        print("2. Train new model")
        choice_mod = input("\nChoose model settings: ")
        
        if (choice_mod == "1"):
            path_mod = input("Input path to model: ")
            model = keras.models.load_model(path_mod)
            clear()
            print(bar)
            print(f"Predicting validation data using model: {path_mod}")

            if choice_set == "1":
                X_train, y_train, X_test, y_test = create_datasets("mnist")

            elif choice_set == "2":
                X_train, y_train, X_test, y_test = create_datasets("mnistfashion")

            loss, acc = model.evaluate(X_test, y_test, verbose = 2)
            print(f"Running model {model} on {dataset} dataset")
            print(f"Accuracy: {acc*100} %")

            check = False

        elif (choice_mod == "2"):
            if choice_set == "1":
                X_train, y_train, X_test, y_test = create_datasets("mnist")

            elif choice_set == "2":
                X_train, y_train, X_test, y_test = create_datasets("mnistfashion")
            
            start = time.time()
            clear()
            print(bar)
            print(f"Training model on {dataset} dataset")
            model = CNN_model(X_train, y_train, X_test, y_test)
            if choice_set == "1":
                model.compile("model_mnist")

            elif choice_set == "2":
                model.compile("model_mnist_fashion")

            model.train()
            end = time.time()
            print(f"Time: {end-start} seconds")
            check = False 

#---RUN CODE-------------+
if __name__ == "__main__":
    main()

