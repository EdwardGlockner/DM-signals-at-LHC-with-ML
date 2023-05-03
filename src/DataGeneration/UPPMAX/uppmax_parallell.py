import subprocess
import numpy as np
import os

""" 
Script to send in jobs to parallell run them in UPPMAX. Run from home folder where MG5_..., SPheno_.. directories are. 
"""

dir = os.getcwd()
script_path =os.path.join(dir, 'scripts_automate_MG5', 'uppmax_run.sh')
print(dir)

# Process
model_name = "neutrino"

# Generate mass list
lower_bound=150
upper_bound=1000
num_elements=10001

if model_name=="neutralino":
    lower_bound=lower_bound
    upper_bound=upper_bound

elif model_name=="neutrino":
    lower_bound=35000.000
    upper_bound=upper_bound**2

spacing=(upper_bound - lower_bound) / (num_elements - 1)
print(spacing)

# List of all mass, first one becomes a different size. 

mass_list = [format(lower_bound + (i * spacing), '.3f') for i in range(num_elements)]
mass_list_split = np.array_split(mass_list, 100)

# Send masses as an input to bash script

for mass_list in mass_list_split:
    # Create an argument to start bash script with mass_list as input
    args = [script_path] + [str(mass) for mass in mass_list] #for UPPMAX add ['sbatch', './test.sh'] for first argument. '/home/max/scripts_automate_MG5/uppmax_run.sh'
    print('New mass list')
    # Submit to bash script
    subprocess.run(args)

