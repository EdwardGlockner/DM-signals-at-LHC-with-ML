#!/bin/bash


current_dir=$(pwd)

# Model name
model_name="neutralino"


# Define file patha
LesHouches_file_path="$current_dir/SPheno-4.0.5/input/LesHouches.in.MSSM_low"

MG5_param_card_path="$current_dir/MG5_aMC_v3_4_2/MSSMmonojet/Cards/param_card.dat"


# Create a mass array
lower_bound=150
upper_bound=1000
spacing=50

#num_elements=$(( (upper_bound - lower_bound) / spacing + 1))
num_elements=10

# Create a mass array
masses=()

#for (( i=0; i<$num_elements; i++ ))
#do 
#	add_mass=$(( lower_bound + i * spacing ))
#	# echo "New mass: $add_mass"
#	masses+=($add_mass)
#done


add_mass=500
for (( i=0; i<num_elements; i++ ))
do 
	masses+=($add_mass)
done

echo "Number of elements: $num_elements"
echo "Masses: ${masses[@]}"

for (( i=0; i<num_elements; i++ )) 					#mass in "${masses[@]}"
do
	# Delete row number for replacement, for DM candidate mass change
	sed -i '18d' $LesHouches_file_path


	# New mass to change it with
	# For muinut = 2E+02 -> chi_1 = 2.05E+02 --> Lower limit
	# For muinut = 3E+02 -> chi_1 = 3.07E+02
	# For muinut = 6E+02 -> chi_1 = 6.12E+02
	# For muinut = 1E+03 -> chi_1 = 1.02E+03 --> Upper limit

	# Insert new line between line 2 and 3, i.e previous deleted row
	echo "Mass: ${masses[i]}"
	#new_mass="$mass"
	sed -i "17a\ \23 ${masses[i]}" $LesHouches_file_path


	# Run SPheno from bash script
	${current_dir}/SPheno-4.0.5/bin/SPhenoMSSM ${current_dir}/SPheno-4.0.5/input/LesHouches.in.MSSM_low


	# Copy the mass of chi_1 and store it in a .csv file
	mass_LSP=$(awk -v row=364 'NR==row {print $2}' ${current_dir}/SPheno.spc.MSSM)
	echo "mass of LSP: $mass_LSP"


	# Store mass in Mass_DM.txt file
	echo "$mass_LSP" >> ${current_dir}/Storage_data/Mass_output/Mass_DM.csv


	# Copy the output file, SPheno.spc.MSSM to param_card.dat in MG5
	cp -f ${current_dir}/SPheno.spc.MSSM $MG5_param_card_path 


	# Run MG5 from terminal
	.${current_path}/MG5_aMC_v3_4_2/MSSMmonojet/bin/generate_events -f #--nb_core=12


	# Find the directory of output diagrams
	search_directory="${current_dir}/MG5_aMC_v3_4_2/MSSMmonojet/HTML/run_01"

	output_folder_path=$(find "$search_directory" -type d -name "tag_*" -print -quit)

	echo "Output folder found: $output_folder_path"


	# Cut and Copy the histograms to a storage directory
	# storage_dir="${current_dir}/Storage_data"
	x=46
	y=10
	width=646
	height=420
 	
	substring="selection"
	src_dir="$output_folder_path/Output/PDF/MadAnalysis5job_0"
	
	for file in "$src_dir"/*;
       	do
		if [[ "$file" == *"$substring"* ]]; then
			convert "$file" -crop "${width}x${height}+${x}+${y}" "$file"
			convert "$file" -colorspace Gray "$file"
		fi
	done

	cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_0.png $current_dir/Storage_data/MSSMmonojet/ALL_png/${i}_${model_name}_MET.png
	cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_1.png $current_dir/Storage_data/MSSMmonojet/ALL_png/${i}_${model_name}_ETA1.png
	cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_2.png $current_dir/Storage_data/MSSMmonojet/ALL_png/${i}_${model_name}_ETA2.png
	cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_3.png $current_dir/Storage_data/MSSMmonojet/ALL_png/${i}_${model_name}_PT1.png
	cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_4.png $current_dir/Storage_data/MSSMmonojet/ALL_png/${i}_${model_name}_PT2.png


	# Delete the folders after each run. Remove run_01 in events and tag_ in HTML/run_01
	rm -rf ${current_dir}/MG5_aMC_v3_4_2/MSSMmonojet/Events/run_01
	find /home/max/MG5_aMC_v3_4_2/MSSMmonojet/HTML/run_01 -type d -name "*tag_*" -print0 | xargs -0 rm -rf
	
done



# TODO after copy the histograms
# 1. Take the DM candidate mass and label the png of what features they have /PET, MET, etc.


