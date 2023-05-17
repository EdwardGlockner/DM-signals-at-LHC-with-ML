#---IMPORTS----------+
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

#---FUNCTIONS--------+

def create_sets_from_csv(*file_paths):
    print("Reading csv files...")
    """
    asdfasdf

    @arguments:
        folder_path: <string> Full path to the csv file
    @returns:

    """
    dfs = []
    for file_path in file_paths:
        df_temp = pd.read_csv(file_path, header=None)
        df_temp = pd.DataFrame({
            'mass': df_temp.iloc[:, 0],
            'model': df_temp.iloc[:, 1],
            'eta': df_temp.iloc[:, 2:47].values.tolist(),
            'pt': df_temp.iloc[:, 47:92].values.tolist(),
            'tet': df_temp.iloc[:, 92:].values.tolist()
            })

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Remove rows that have NaN values
    df['eta'] = df['eta'].apply(lambda x: [val for val in x if not np.isnan(val)])
    df['pt'] = df['pt'].apply(lambda x: [val for val in x if not np.isnan(val)])
    df['tet'] = df['tet'].apply(lambda x: [val for val in x if not np.isnan(val)])

    df = df[df['eta'].str.len() > 0]
    df = df[df['pt'].str.len() > 0]
    df = df[df['tet'].str.len() > 0]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns",None)
    # Create separate arrays
    masses = df["mass"].values
    models = df["model"].values
    eta_vals = df["eta"].values
    pt_vals = df["pt"].values
    tet_vals = df["tet"].values
    
    # Convert the elements into numpy.ndarrays
    eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
    pt_vals = np.array([np.array(sublist) for sublist in pt_vals])
    tet_vals = np.array([np.array(sublist) for sublist in tet_vals])
    # Concatenate into three channels
    input_data  = np.concatenate([eta_vals[..., np.newaxis], pt_vals[..., np.newaxis], tet_vals[..., np.newaxis]], axis=2)
    return input_data, masses, models


def data_augmentation_cl(X_train, y_train, augment_size):
    """

    """
    print("Running data augmentation...")
    # Reshape the input data
    X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # Add an additional dimension

    # Create an instance of ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,  # Random rotation in the range of [-10, 10] degrees
        width_shift_range=0.1,  # Random horizontal shift by 0.1 of the total width
        height_shift_range=0.1,  # Random vertical shift by 0.1 of the total height
        zoom_range=0.15,  # Random zoom by 0.1
        channel_shift_range=2
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


def dat_augmentation_re(X_train_hist, y_train, augment_size):
    return


def shuffle_and_create_sets(X_data, labels, targets, random_seed = 13, print_shapes = False):
    print("Creating sets and shuffling data...")
    """
    Shuffles all the images and labels/targets and creates three datasets with a ratio of 0.64:0.16:0.20.
    The training and testing sets are passed to the model in order to train it, while the validation set
    is an out of sample test, used for performance metrics.

    @arguments:
        X_data:    <numpy.ndarray> Array of all the input data stored in 3 channels.
        labels: <numpy.ndarray> Array of class labels for classification
        targets: <numpy.ndarray> Array of target values for regression
        random_seed:     <int> Random seed for the shuffling
        print_shapes:    <bool> Whether to print the shapes of the sets or not
    @returns:
        X_train: <numpy.ndarray> Contains 64 % of the data
        y_train: <numpy.ndarray> Contains 64 % of the data
        X_test:  <numpy.ndarray> Contains 16 % of the data 
        y_test:  <numpy.ndarray> Contains 16 % of the data
        X_val:   <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
        y_val:   <numpy.ndarray> Used for validation after the model is trained. Contains 20 % of the data
    """
    # Check if all arrays have the same length as the first array
    same_length = np.all(np.array([len(X_data), len(labels), len(targets)]) == len(X_data))

    if not same_length: # Return empty sets if error
        print("Error in function <shuffle_and_create_sets>. Arrays are not the same size.")
        return [], [], [], [], [], []
    
    # Randomly shuffle the sets
    np.random.seed(random_seed)
    permutation = np.random.permutation(len(X_data))

    X_data_shuffled = X_data[permutation]
    labels_shuffled = labels[permutation]
    targets_shuffled = targets[permutation]

    # Create training, validation and testing sets using 0.64:0.16:0.20
    tot_len = len(X_data)
    first_split = int(tot_len * 0.70)
    second_split = int(first_split + tot_len * 0.16)
    
    # cl for classification, re for regression
    X_train, X_val, X_test = X_data_shuffled[0:first_split], X_data_shuffled[first_split:second_split], X_data_shuffled[second_split:]
    y_train_cl, y_val_cl, y_test_cl = labels_shuffled[0:first_split], labels_shuffled[first_split:second_split], labels_shuffled[second_split:]
    y_train_re, y_val_re, y_test_re = targets_shuffled[0:first_split], targets_shuffled[first_split:second_split], targets_shuffled[second_split:]
    
    # Use data augmentation
    augmented_X_train_cl, augmented_y_train_cl= data_augmentation_cl(X_train, y_train_cl, augment_size=200000)
    augmented_y_train_cl = np.ravel(augmented_y_train_cl)
    print(y_train_cl.shape, "before")
    X_train_cl = np.concatenate((X_train, augmented_X_train_cl))
    y_train_cl_aug = np.concatenate((y_train_cl, augmented_y_train_cl))
    print(y_train_cl_aug.shape, "after")

    X_train = np.array([np.expand_dims(sample, axis=-1) for sample in X_train])
    X_train_cl = np.array([np.expand_dims(sample, axis=-1) for sample in X_train_cl])
    X_test = np.array([np.expand_dims(sample, axis=-1) for sample in X_test])
    X_val = np.array([np.expand_dims(sample, axis=-1) for sample in X_val])

    return [X_train_cl, y_train_cl_aug, X_test, y_test_cl, X_val, y_val_cl], \
            [X_train, y_train_cl, y_train_re, X_test, y_test_cl, y_test_re, \
            X_val, y_val_cl, y_val_re]


