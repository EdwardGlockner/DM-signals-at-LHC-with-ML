#!/bin/bash
#--------------- For UPPMAX run ------------------
##SBATCH -A name-of-project(ex. naiss2023-22-305)
##SBATCH -p core
##SBATCH -n 1 
##SBATCH -t 12:00:00
##SBATCH -J (jobs, but script in this file)

# Load python module, madgraph OK for python 3.7 or higher
##module load python/3.10.8

current_dir=$(pwd)

python_version="python3.10"

# Run the machine learning model
"$python_version" main.py -r trainval -m first_run
