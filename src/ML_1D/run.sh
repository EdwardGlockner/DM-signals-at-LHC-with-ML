#!/bin/bash
#--------------- For UPPMAX run ------------------
##SBATCH -A name-of-project(ex. naiss2023-22-305)
##SBATCH -p core
##SBATCH -n 1 
##SBATCH -t 12:00:00
##SBATCH -J (jobs, but script in this file)

# Load python module
##module load python/3.10.8

python_version="python3.10"

# Run the machine learning model
$python_version main.py --run trainval --type cl --name ForReport 
