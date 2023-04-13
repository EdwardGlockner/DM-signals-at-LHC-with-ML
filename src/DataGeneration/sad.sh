#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
echo "Running main script ..."
echo " "

filename="testing.txt"

read -p "Enter the search string: " search
read -p "Enter the replace string: " replace

if [[ $search != "" && $replace != "" ]]; then
   sed -i " " "s/$search/$replace/" $filename
fi

