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
	NPROC=$(nproc)
	echo "number of cores: $NPROC"

	# Go to UPPMAX project directory to run 
	#cd ...
	# Should I copy /home/max/... to the project directory to run the files?



# Start the run in the /home/max/... directory to run bash script
current_dir=$(pwd)


#-------------- Constant parameters -----------------

# Model name
model_name="neutralino" #neutrino #something #something2
signature="jet" 		#z

# Masses boundary and number of elements
lower_bound=150
upper_bound=1000
num_elements=101

# Dimensions of the histograms for cut
x=46
y=10
width=646
height=420


#--------------------- Run script --------------------

# Before running the script. 
# 1. Run madgraph and launch one run (output "MSSM_${model_name}_${signature}") with the preferred settings (Pythia8 [on off], Delphes [on off], Madanalysis5 [on off], correct run_card.dat, 
# delphes_card.dat, madanalysis5_parton_card.dat) param_card.dat is solved within this bash script. For neutrino process, correct the invisible particles in delphes_card.dat at ECAL, HCAL and define
# invisible particles in madanalysis5_card.dat correctly (+define sve sve~).
# 2. Prepare the storage directory named as ("Storage_data") and create the folder ("Mass_output", "Model_output", "MSSM_${model_name}_${signature}"). Within "MSSM_${model_name}_${signature}" 
# create a directory named as ("raw_images")
# 3. run script in project directory where MG5_aMC_v3_4_2, SPHeno-4.0.5, Storage_data etc are with the command
#		bash scripts_automate_MG5/automate_data.sh
#


# Define file paths and define lower and upper bound of masses to LesHouches files

LesHouches_file_path="$current_dir/SPheno-4.0.5/input/LesHouches.in.MSSM_${model_name}_${signature}"

MG5_param_card_path="$current_dir/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/Cards/param_card.dat"

if [ $model_name == "neutralino" ] 
then
	lower_bound=$lower_bound
	upper_bound=$upper_bound

	echo $lower_bound
	echo $upper_bound

elif [ $model_name == "neutrino" ]
then
	upper_bound=$((upper_bound**2))
	#lower_bound=$((lower_bound**2))

	# To obtain approximate 150 GeV of LSP after SPHeno, lower bound has to be modified
	lower_bound=35000.000
elif [ $model_name == "something"]
then
	echo "insert file path"
fi	

spacing=$(echo "scale=3; ($upper_bound - $lower_bound) / ($num_elements - 1)" | bc)
echo $spacing

# Create a mass array
masses=()

for (( i=0; i<$num_elements; i++ ))
do 
	add_mass=$(echo "$lower_bound + ($i * $spacing)" | bc)
	masses+=($add_mass)
	#echo "Masses: ${masses[$i]}"
done

echo "Number of elements: $num_elements"
echo "Masses: ${masses[100]}"


# Iterate through all of masses, run madgraph and store output histograms in the corresponding dir

for (( i=0; i<num_elements; i++ )) 	
do
	echo "Mass: ${masses[i]}"

	if [ "$model_name" == "neutralino" ]
	then	

		# Delete row where to change the mass and then insert new row at same place with new mass
		sed -i '18d' $LesHouches_file_path
		sed -i "17a\ \23 ${masses[${i}]}" $LesHouches_file_path

		# Run SPheno
		${current_dir}/SPheno-4.0.5/bin/SPhenoMSSM $LesHouches_file_path

		# Copy the mass of chi_1
		mass_LSP=$(awk -v row=364 'NR==row {print $2}' ${current_dir}/SPheno.spc.MSSM)

	elif [ "$model_name" == "neutrino" ]
	then
		sed -i '114d' $LesHouches_file_path
		sed -i "113a\ \1 1 ${masses[i]}" $LesHouches_file_path

		# Run SPheno
		${current_dir}/SPheno-4.0.5/bin/SPhenoMSSM $LesHouches_file_path

		# Copy the mass of chi_1
		mass_LSP=$(awk -v row=340 'NR==row {print $2}' ${current_dir}/SPheno.spc.MSSM)

	elif [ "$model_name" == "something" ]
	then
		echo "something"
	fi


	echo "mass of LSP: $mass_LSP"

	### From here I can run outside of if statement!!!

	# Store mass in a .csv file
	echo "$mass_LSP" >> ${current_dir}/Storage_data/Mass_output/Mass_DM_${model_name}.csv

	if [ $model_name == "neutralino" ] 
	then
		echo "0" >> ${current_dir}/Storage_data/Model_output/Model_DM_${model_name}.csv
	elif [ $model_name == "neutrino" ]
	then
		echo "1" >> ${current_dir}/Storage_data/Model_output/Model_DM_${model_name}.csv
	elif [ $model_name == "something"]
	then
		echo "insert file path"
	fi	

	

	# Copy the output file, SPheno.spc.MSSM to overwrite param_card.dat in MG5
	cp -f ${current_dir}/SPheno.spc.MSSM $MG5_param_card_path 

	for (( j=0; j<2; j++ ))
	do
		# Run MG5 from terminal
		.${current_path}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/bin/generate_events -f --nb_core=${NPROC}

		# Find the directory of output diagrams
		search_dir="${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/HTML/run_01"

		output_folder_path=$(find "$search_dir" -type d -name "tag_*" -print -quit)

		echo "Output folder found: $output_folder_path"

		substring="selection"
		histogram_dir="$output_folder_path/Output/PDF/MadAnalysis5job_0"
		
		for file in "$histogram_dir"/*;
		do
			if [[ "$file" == *"$substring"* ]]; then
				convert "$file" -crop "${width}x${height}+${x}+${y}" "$file"
				convert "$file" -colorspace Gray "$file"
			fi
		done

		cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_0.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${i}_${j}_${model_name}_${signature}_MET.png
		cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_1.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${i}_${j}_${model_name}_${signature}_ETA.png
		cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_2.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${i}_${j}_${model_name}_${signature}_PT.png

		# Delete the folders after each run. Remove run_01 in events and tag_ in HTML/run_01
		rm -rf ${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/Events/run_01
		find ${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/HTML/run_01 -type d -name "*tag_*" -print0 | xargs -0 rm -rf
	done
done
