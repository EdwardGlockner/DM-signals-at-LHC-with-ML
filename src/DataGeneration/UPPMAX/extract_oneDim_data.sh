#!/bin/bash

# This script extracts the ETA, MET and PT into one_dim_data.csv
# If this file does not exist, it will be created on first run

#PATH TO SCRIPTS, MAKE SURE TO CHANGE PATH TO CORRECT ONE
path="\home\max\MG5_aMC_v3_4_2\MSSM_neutrino_jet\HTML\run_29\tag_1_MA5_PARTON_ANALYSIS_analysis1\Output\Histos\MadAnalysis5job_0\\"

# read the contents of the python script into a variable
sel0=$(cat $path"selection_0.py")
sel1=$(cat $path"selection_1.py")
sel2=$(cat $path"selection_2.py")

# Extract contents of numpy.array(...) on line 20
eta=$(echo "$sel0" | sed -n '20p' | awk -F '[()]' '{print $2}')
pt=$(echo "$sel0" | sed -n '20p' | awk -F '[()]' '{print $2}')
met=$(echo "$sel0" | sed -n '20p' | awk -F '[()]' '{print $2}')

# Save to csv
echo "$eta,$met,$pt" >> one_dim_data.csv