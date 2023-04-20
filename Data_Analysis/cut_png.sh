#!/bin/bash

#692 x 472 pixlar 

# 646 x 430 pixlar 

current_dir=$(pwd)

input_image="${current_dir}/Storage_data/selection_0.png"

output_image="${current_dir}/Storage_data/selection_0_new.png"

x=46
y=10
width=646
height=425

convert "$input_image" -crop "${width}x${height}+${x}+${y}" "$output_image"

echo "Image has been cropped"
