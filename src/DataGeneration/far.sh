#!/bin/bash
echo "Running main script ..."
echo " "

# Find the line number which should be removed
read -p "Enter the file which should be modified: " file_name
read -p "Enter the line number that should be replaced: " line_number
read -p "Enter the new line: " new_line

# Use sed to remove the specified line from the file
sed -i '' "${line_number}d" "$file_name"

# Use sed to insert the new line
sed -i '' "${line_number}i\\
${new_line}\\
" "$file_name"

echo " "
echo "Script done running!"
echo " "
