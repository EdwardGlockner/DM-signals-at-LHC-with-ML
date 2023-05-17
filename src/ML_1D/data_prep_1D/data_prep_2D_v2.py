#---IMPORTS----------+
import numpy as np
import pandas as pd
import os
import sys

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")
print(dirname)

#---FUNCTIONS--------+
def load_sets():
    """

    """
    test_input = dirname + "/Storage_data/data_jet/test_input.csv"
    test_mass = dirname + "/Storage_data/data_jet/test_mass.csv"
    test_model = dirname + "/Storage_data/data_jet/test_model.csv"
    val_input = dirname + "/Storage_data/data_jet/validation_input.csv"
    val_mass = dirname + "/Storage_data/data_jet/validation_mass.csv"
    val_model = dirname + "/Storage_data/data_jet/validation_model.csv"
    train_input = dirname + "/Storage_data/data_jet/train_input.csv"
    train_mass = dirname + "/Storage_data/data_jet/train_mass.csv"
    train_model = dirname + "/Storage_data/data_jet/train_model.csv"
    
    X_train = df_input(train_input)
    X_test = df_input(test_input)
    X_val = df_input(val_input)

    mass_train = df_mass_model(train_mass)
    mass_test = df_mass_model(test_mass)
    mass_val = df_mass_model(val_mass)

    model_train = df_mass_model(train_model)
    model_test = df_mass_model(test_model)
    model_val = df_mass_model(val_model)

    cl_dataset = [X_train, model_train, X_test, model_test, X_val, model_val]
    re_dataset = [X_train, model_train, mass_train, X_test, model_test, mass_test, \
            X_val, model_val, mass_val]

    return cl_dataset, re_dataset
   
def df_input(file_path):
    """

    """
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'eta': df.iloc[:, 0:45].values.tolist(),
        'pt': df.iloc[:, 45:90].values.tolist(),
        'tet': df.iloc[:, 90:].values.tolist()
    })

    eta_vals = df["eta"].tolist()
    pt_vals = df["pt"].tolist()
    tet_vals = df["tet"].tolist()

    eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
    pt_vals = np.array([np.array(sublist) for sublist in pt_vals])
    tet_vals = np.array([np.array(sublist) for sublist in tet_vals])
    # Concatenate into three channels
    input_data  = np.concatenate([eta_vals[..., np.newaxis], pt_vals[..., np.newaxis], tet_vals[..., np.newaxis]], axis=2)
    return input_data

def df_mass_model(file_path):
    """

    """
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'values': df.iloc[:,0]
    })

    values = df['values'].values
    return values






