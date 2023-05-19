import pandas as pd

df = pd.read_csv('./train_mass.csv')
# Check for NaN values
has_nan = df.isna().any().any()

if has_nan:
    print("The DataFrame contains NaN values.")
else:
    print("The DataFrame does not contain any NaN values.")

rows_with_nan = df[df.isna().any(axis=1)]

# Print the index of rows with NaN values
print(rows_with_nan.index)
