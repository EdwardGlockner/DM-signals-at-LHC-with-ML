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

    # Go to UPPMAX project directory to not run in "cluster" that won't save data.
    # cd /back_to_project_dir

# Get available cores on cluster
NPROC=$(nproc)
echo "number of cores: $NPROC"

current_dir=$(pwd)

#-------------- Constant parameters -----------------

# Model name
model_name="neutrino" #neutrino #something #something2
signature="jet" 		#z

# Dimensions of the histograms for cut
x=46
y=10
width=646
height=420

#--------------------- Run script (parallell jobs) --------------------

# Define file paths for LesHouches and param_card paths

LesHouches_file_path="$current_dir/SPheno-4.0.5/input/LesHouches.in.MSSM_${model_name}_${signature}"

MG5_param_card_path="$current_dir/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/Cards/param_card.dat"

# Create a mass list from python script. Send input in subprocess.run(args) where args is ['sbatch', './uppmax_run.sh', '150', '155', ... , '1000']

mass_list=("$@")

echo "Mass_list = ${mass_list[@]}"

# Iterate through all of masses, run madgraph and store output histograms in the corresponding dir

for mass in "${mass_list[@]}"
do
    if [ "$model_name" == "neutralino" ]
    then	
        # Delete row where to change the mass and then insert new row at same place with new mass
        sed -i '18d' $LesHouches_file_path
        sed -i "17a\ \23 ${mass}" $LesHouches_file_path

        # Run SPheno
        ${current_dir}/SPheno-4.0.5/bin/SPhenoMSSM $LesHouches_file_path

        # Copy the mass of chi_1
        mass_LSP=$(awk -v row=364 'NR==row {print $2}' ${current_dir}/SPheno.spc.MSSM)

    elif [ "$model_name" == "neutrino" ]
    then
        # Delete row where to change the mass and then insert new row at same place with new mass
        sed -i '114d' $LesHouches_file_path
        sed -i "113a\ \1 1 ${mass}" $LesHouches_file_path

        # Run SPheno
        ${current_dir}/SPheno-4.0.5/bin/SPhenoMSSM $LesHouches_file_path

        # Copy the mass of chi_1
        mass_LSP=$(awk -v row=340 'NR==row {print $2}' ${current_dir}/SPheno.spc.MSSM)

    elif [ "$model_name" == "something" ]
    then
        echo "something"
    fi

    echo "mass of LSP: $mass_LSP"
    mass_LSP_to_image=$(printf "%.3f" $mass_LSP)
    echo "rounded LSP: $mass_LSP_to_image"


    # Copy the output file, SPheno.spc.MSSM to overwrite param_card.dat in MG5
    cp -f ${current_dir}/SPheno.spc.MSSM $MG5_param_card_path

    # Run MG5 from terminal
    .${current_path}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/bin/generate_events -f --nb_core=${NPROC}

    # set run_mode 2 -> Done in mg5.configuration.txt (setting to not run more cores than given on UPPMAX)

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

    ##--------------------------------------
    path="$output_folder_path/Output/Histos/MadAnalysis5job_0/"

    # read the contents of the python script into a variable
    sel0=$(cat $path"selection_0.py")
    sel1=$(cat $path"selection_1.py")
    sel2=$(cat $path"selection_2.py")

    # Extract contents of numpy.array(...) on line 20	
    eta=$(echo "$sel0" | sed -n '20p' | awk -F '[\\[\\]]' '{print $2}')
    pt=$(echo "$sel0" | sed -n '20p' | awk -F '[\\[\\]]' '{print $2}')
    met=$(echo "$sel0" | sed -n '20p' | awk -F '[\\[\\]]' '{print $2}')

    # Save to csv
    echo "$mass_LSP_to_image, $eta" >> $current_dir/Storage_data/MSSM_${model_name}_${signature}/norm_amp_array/array_ETA.csv
    echo "$mass_LSP_to_image, $pt" >> $current_dir/Storage_data/MSSM_${model_name}_${signature}/norm_amp_array/array_PT.csv
    echo "$mass_LSP_to_image, $met" >> $current_dir/Storage_data/MSSM_${model_name}_${signature}/norm_amp_array/array_MET.csv	
    #
    ##--------------------------------------

    cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_0.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${mass_LSP_to_image}_${model_name}_${signature}_ETA.png
    cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_1.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${mass_LSP_to_image}_${model_name}_${signature}_PT.png
    cp $output_folder_path/Output/PDF/MadAnalysis5job_0/selection_2.png $current_dir/Storage_data/MSSM_${model_name}_${signature}/raw_images/${mass_LSP_to_image}_${model_name}_${signature}_MET.png

    # Delete the folders after each run. Remove run_01 in events and tag_ in HTML/run_01
    rm -rf ${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/Events/run_01
    rm -rf ${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/HTML/run_01
    #find ${current_dir}/MG5_aMC_v3_4_2/MSSM_${model_name}_${signature}/HTML/run_01 -type d -name "*tag_*" -print0 | xargs -0 rm -rf
done
