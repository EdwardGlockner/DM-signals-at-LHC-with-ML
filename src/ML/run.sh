#!/bin/bash
	#--------------- For UPPMAX run ------------------
	##!/bin/bash -l

	##SBATCH -A name-of-project(ex. naiss2023-22-305)
	##SBATCH -p core
	##SBATCH -n 1 
	##SBATCH -t 12:00:00
	##SBATCH -J (jobs, but script in this file)

	# Load python module, madgraph OK for python 3.7 or higher
	#module load python/3.8.7

	# Get cores of machine
	#NPROC=$(nproc)
	#echo "number of cores: $NPROC"
    #run_mode = 2??
	# Go to UPPMAX project directory to run 
	#cd ...
	# Should I copy /home/max/... to the project directory to run the files?

python3.10 main_training.py
