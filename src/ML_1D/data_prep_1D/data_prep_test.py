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
def load_sets(signature, model_type):
    """
    asdfasdf

    @arguments:
        signature: <string>
        model_type: <string>
    @returns:
        None
    """
    if signature == "z":
        test_input = dirname + "Storage_data/z_data/Testing/monoz_input_testing.csv"
        test_mass = dirname + "Storage_data/z_data/Testing/monoz_mass_testing.csv"
        test_model = dirname + "Storage_data/z_data/Testing/monoz_model_testing.csv"
        
        X_test = df_input_z(test_input)
        mass_test = df_mass_model(test_mass)
        model_test = df_mass_model(test_model)
        model_test = np.where(model_test == 2, 0, np.where(model_test == 3, 1, model_test))
    
    else: # jet

        test_input = dirname + "Storage_data/jet_data/Testing/jet_input_testing.csv"
        test_mass = dirname + "Storage_data/jet_data/Testing/jet_mass_testing.csv"
        test_model = dirname + "Storage_data/jet_data/Testing/jet_model_testing.csv"
    
        X_test = df_input_jet(test_input)
        mass_test = df_mass_model(test_mass)
        model_test = df_mass_model(test_model)
    
    if model_type == "cl":
        return [X_test, model_test]
    elif model_type == "re":
        return [X_test, model_test, mass_test]
    else:
        print("Invalid model_type in <load_sets>")
        return 


def df_input_jet(file_path):
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


def df_input_z(file_path):
    """

    """
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'eta': df.iloc[:, 0:45].values.tolist(),
        'pt': df.iloc[:, 45:90].values.tolist(),
        'mt_met': df.iloc[:, 90:135].values.tolist(),
        'tet': df.iloc[:, 135:].values.tolist()
    })

    eta_vals = df["eta"].tolist()
    pt_vals = df["pt"].tolist()
    mt_met_vals = df["mt_met"].tolist()
    tet_vals = df["tet"].tolist()

    eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
    pt_vals = np.array([np.array(sublist) for sublist in pt_vals])
    mt_met_vals = np.array([np.array(sublist) for sublist in mt_met_vals])
    tet_vals = np.array([np.array(sublist) for sublist in tet_vals])

    # Concatenate into four channels
    input_data = np.concatenate([eta_vals[..., np.newaxis], pt_vals[..., np.newaxis] ,\
            mt_met_vals[..., np.newaxis], tet_vals[..., np.newaxis]], axis=2)

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

