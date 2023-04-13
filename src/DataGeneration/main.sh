#!/bin/bash


# For loop
#    rewrite spheno input params. Randomly vary mass between 150 - 500/750/1000
#    run spheno and copy the output and use it as param_card
#    run MG5 from the command prompt (./generate_events.py (-s -f))
#    Copy the output generated from MA5, and delete all the other files to save storage
#    Run MA5 and store the plots + additional data (such as mass)
#
#
#main.sh
#!/bin/bash

# This is the main file running the HPC on UPPMAX
# The steps performed in this file is:
#       1. Rewriting the SPheno input parameters. Randomly varying the mass between 150 - 750
#       2. Running SPheno and copying the output, renaming it and using it as param_card.dat
#       3. Running MG5 from the command prompt
#       4. Copying the output generated from MA5, and deleting all the other files in order to save storage
#       5. Running MA5 and storing the plots, and additional data (such as mass)


# Main for loop
#path_LesHouches="/home/edward/Desktop/Thesis/SPheno-4.0.5/input/LesHouces.in.MSSM"
path_storage="~/Desktop/Thesis/storage_folder/storage_folder"

for i in {1..2}
do
        # Here i need to modify the parameters of the input to SPheno.
        # I also need to randomly modify the masses between 150 and 750
        # Running SPheno
        ~/Desktop/Thesis/SPheno-4.0.5/bin/SPhenoMSSM ~/Desktop/Thesis/SPheno-4.0.5/input/LesHouches.in.MSSM
        # Copying the output, renaming it and using it as a param_card.dat
        rm -r ~/Desktop/Thesis/MG5_aMC_v3.4.2/MG5_aMC_v3_4_2/MSSMmonojet/Cards/param_card.dat
        cp ~/Desktop/Thesis/SPheno-4.0.5/SPheno.spc.MSSM ~/Desktop/Thesis/MG5_aMC_v3.4.2/MG5_aMC_v3_4_2/MSSMmonojet/Cards/param_card.dat 
                
        # Running MG5 from the terminal
        ~/Desktop/Thesis/MG5_aMC_v3.4.2/MG5_aMC_v3_4_2/MSSMmonojet/bin/generate_events -f

        # Copying the output generated from MA5 and storing the plots
        substring="selection"
        src_dir = "~/Desktop/Thesis/MG5_aMC_v3.4.2/MG5_aMC_v3_4_2/MSSMmonojet/HTML/run_03/tag_1_MA5_PARTON_ANALYSIS_analysis1/Output/PDF/MadAnalysis5job_0"
        for file in "$src_dir"/*; do
                if [[ "$file" == *"$substring"* ]]; then
                        mv "$file" "$dest_dir"
                fi
        done
done
