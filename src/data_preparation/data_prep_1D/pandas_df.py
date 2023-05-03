import pandas as pd
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("../raw_data_all.csv", header=None)

# Create a new dataframw with columns, [mass, model, eta, pt, met]
# model values = {neutralino_jet, 0; neutrino_jet, 1}

df =pd.DataFrame({
    'mass': df.iloc[:, 0],
    'model': df.iloc[:, 1],
    'eta': df.iloc[:, 2:52].values.tolist(),
    'pt': df.iloc[:, 52:102].values.tolist(),
    'met': df.iloc[:, 102:].values.tolist()
})

masses = df["mass"].values
models = df["model"].values
eta_vals = df["eta"].values
pt_vals = df["pt"].values
met_vals = df["met"].values

eta_vals = np.array([np.array(sublist) for sublist in eta_vals])
pt_vals = np.array([np.array(sublist) for sublist in pt_vals])
met_vals = np.array([np.array(sublist) for sublist in met_vals])
