import pandas as pd
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("raw_data_all.csv", header=None)

# Create a new dataframw with columns, [mass, model, eta, pt, met]
# model values = {neutralino_jet, 0; neutrino_jet, 1}

df =pd.DataFrame({
    'mass': df.iloc[:, 0],
    'model': df.iloc[:, 1],
    'eta': df.iloc[:, 2:52].values.tolist(),
    'pt': df.iloc[:, 52:102].values.tolist(),
    'met': df.iloc[:, 102:].values.tolist()
})

print(df.head())
