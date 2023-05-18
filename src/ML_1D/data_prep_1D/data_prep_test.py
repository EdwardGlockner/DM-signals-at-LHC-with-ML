#---IMPORTS----------+
import numpy as np
import pandas as pd
import os
import sys
from keras.preprocessing.image import ImageDataGenerator

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")

#---FUNCTIONS--------+
def load_sets(model_type):
    """
    asdfasdf

    @arguments:
        model_type: <string>
    @returns:
        None
    """
    test_input = dirname + "/Storage_data/data_jet/test_input.csv"
    test_mass = dirname + "/Storage_data/data_jet/test_mass.csv"
    test_model = dirname + "/Storage_data/data_jet/test_model.csv"
    
    X_test = df_input(test_input)
    mass_test = df_mass_model(test_mass)
    model_test = df_mass_model(test_model)
    
    if model_type == "cl":
        return [X_test, model_test]
    elif model_type == "re":
        return [X_test, model_test, mass_test]
    else:
        print("Invalid model_type in <load_sets>")
        return 


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

