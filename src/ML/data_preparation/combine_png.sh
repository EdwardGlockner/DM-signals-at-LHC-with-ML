#!/bin/bash

 current_dir=$(pwd)

 input_directory="${current_dir}/../raw_data/images"
 output_directory="${current_dir}/../processed_data/images"


 #convert +append ${input_directory}/*.png ${output_directory}/combined.png


 # Extract the highest index in directory

 highest_index=$(ls ${current_dir}/Storage_data/MSSMmonojet/MET/*_neutralino_MET.png | \
 		awk '{gsub(".*/|_neutralino_MET.png","");print}' | sort -n  | tail -1 )

 echo "Highest index: $highest_index"

 for (( i=0; i<highest_index; i++ ))
 do 	
 	convert +append ${output_directory}/ALL_png/${i}*.png ${output_directory}/Combined/combined_${i}.png
 done

 # Convert all of the images indices [0 to 17] from the folders and combine them together

 echo "Combined PNG image created at ${output_directory}/combined.png"

