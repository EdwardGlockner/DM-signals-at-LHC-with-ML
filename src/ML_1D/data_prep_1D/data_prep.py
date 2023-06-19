#---IMPORTS----------+
import numpy as np
import pandas as pd
import os

#---FIXING DIRNAME-------+
dirname = os.getcwd()
dirname = dirname.replace("src/ML_1D","")

#---FUNCTIONS--------+
def load_sets(signature, aug=False):
    """
    asdfasdf

    @arguments:
        signature: <>
        aug: <bool>
    @returns:
        None
    """
    if signature == "z":
        train_input = dirname + "Storage_data/cut_z_data/Training/monoz_input_training.csv"
        train_mass = dirname + "Storage_data/cut_z_data/Training/monoz_mass_training.csv"
        train_model = dirname + "Storage_data/cut_z_data/Training/monoz_model_training.csv"
        val_input = dirname + "Storage_data/cut_z_data/Validating/monoz_input_validation.csv"
        val_mass = dirname + "Storage_data/cut_z_data/Validating/monoz_mass_validation.csv"
        val_model = dirname + "Storage_data/cut_z_data/Validating/monoz_model_validation.csv"
    
        X_train = df_input_z(train_input)
        X_val = df_input_z(val_input)

        mass_train = df_mass_model(train_mass)
        mass_val = df_mass_model(val_mass)

        model_train = df_mass_model(train_model)
        model_val = df_mass_model(val_model)
        model_train = np.where(model_train == 2, 0, np.where(model_train == 3, 1, model_train))
        model_val = np.where(model_val == 2, 0, np.where(model_val == 3, 1, model_val))

    else: # jet
        train_input = dirname + "Storage_data/cut_jet_data/Training/jet_input_training.csv"
        train_mass = dirname + "Storage_data/cut_jet_data/Training/jet_mass_training.csv"
        train_model = dirname + "Storage_data/cut_jet_data/Training/jet_model_training.csv"
        val_input = dirname + "Storage_data/cut_jet_data/Validating/jet_input_validation.csv"
        val_mass = dirname + "Storage_data/cut_jet_data/Validating/jet_mass_validation.csv"
        val_model = dirname + "Storage_data/cut_jet_data/Validating/jet_model_validation.csv"

        X_train = df_input_jet(train_input)
        X_val = df_input_jet(val_input)

        mass_train = df_mass_model(train_mass)
        mass_val = df_mass_model(val_mass)

        model_train = df_mass_model(train_model)
        model_val = df_mass_model(val_model)
    
    
    if aug:
        augmented_X_train, augmented_model_train = data_aug(X_train, model_train, \
                augment_size = 500)
        augmented_model_train = np.ravel(augmented_model_train)

        cl_dataset = [np.concatenate((X_train, augmented_X_train)), X_val, model_val]
        re_dataset = [X_train, model_train, mass_train, X_val, model_val, mass_val]
        
        return cl_dataset, re_dataset

    else:
        cl_dataset = [X_train, model_train, X_val, model_val]
        re_dataset = [X_train, model_train, mass_train, X_val, model_val, mass_val]

        return cl_dataset, re_dataset


def df_input_jet(file_path):
    """

    """
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'eta': df.iloc[:, 0:30].values.tolist(),
        'tet': df.iloc[:, 30:].values.tolist()
    })

    eta_vals = df["eta"].tolist()
    tet_vals = df["tet"].tolist()

    eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
    tet_vals = np.array([np.array(sublist) for sublist in tet_vals])

    # Concatenate into three channels
    input_data  = np.concatenate([eta_vals[..., np.newaxis], tet_vals[..., np.newaxis]], axis=2)
    return input_data


def df_input_z(file_path):
    """

    """
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'eta': df.iloc[:, 0:40].values.tolist(),
        'pt': df.iloc[:, 40:80].values.tolist(),
        'mt_met': df.iloc[:, 80:120].values.tolist(),
        'tet': df.iloc[:, 120:].values.tolist()
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
    print(file_path)
    print("asdfasdfasdf")
    df = pd.read_csv(file_path, header=None)
    df = pd.DataFrame({
        'values': df.iloc[:,0]
    })

    values = df['values'].values
    return values


def data_aug(X_train, y_train, augment_size):
    """

    """
    print("Running data aug")
    # Reshape the input data
    X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # Add an additional dimension

    # Create an instance of ImageDataGenerator
    datagen = ImageDataGenerator(
        width_shift_range=0.2,  # Random horizontal shift by 0.1 of the total width
        height_shift_range=0.2,  # Random vertical shift by 0.1 of the total height
        zoom_range=0.15,  # Random zoom by 0.1
        #add rotation
    )

    # Generate augmented data
    augmented_data = []
    augmented_labels = []
    for X_batch, y_batch in datagen.flow(X_train_3d, y_train, batch_size=1, shuffle=False):
        augmented_data.append(X_batch.squeeze())  # Remove the additional dimension
        augmented_labels.append(y_batch)
        if len(augmented_data) >= augment_size:
            break

    # Convert augmented data and labels to arrays
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    return augmented_data, augmented_labels


