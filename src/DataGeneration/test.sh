#test.sh Doesnt fucking work
#!/bin/bash


substring="selection"
path_storage="~/Desktop/Thesis/storage_folder"
src_dir="~/Desktop/Thesis/MG5_aMC_v3.4.2/MG5_aMC_v3_4_2/MSSMmonojet/HTML/run_03/tag_1_MA5_PARTON_ANALYSIS_analysis1/Output/PDF/MadAnalysis5job_0"
file "$path_storage"
ls "$src_dir"
for file in "$src_dir"/*"$substring"*; do
        echo "$file"
        if [[ -f "$file" ]]; then
                mv "$file" "$path_storage"
        fi
done
