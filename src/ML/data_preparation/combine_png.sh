#!/bin/bash

# This script is only for a directory with the same model

input_directory=${1}
output_directory=${2}
echo "input_dir: $input_directory"
echo "output_dir: $output_directory"

# Combine the average pngs in a model directory
for (( i=0; i<10; i++ ))
do 
		convert ${input_directory}/${i}*_ETA.png \
			${input_directory}/${i}*_MET.png \
			${input_directory}/${i}*_PT.png +append ${output_directory}/${i}_combined.png
done

