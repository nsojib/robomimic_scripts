# myscript.sh
#!/bin/bash

filepath=$1 
echo "Filepath: $filepath" 

if [ -d "$filepath" ]; then
    echo "The given path is a directory."

    # Add a trailing slash if it's not already present
    if [ "${filepath: -1}" != "/" ]; then
        filepath="$filepath/"
    fi


    specific_file="demo.hdf5"

    if [ -e "$filepath/$specific_file" ]; then
        echo "The directory contains the specific file: $specific_file"
        filepath="$filepath$specific_file"
    else
        echo "The directory does not contain the specific file: $specific_file"
        exit 1
    fi
else
    echo "The given path is not a directory."
    exit 1
fi
 
directory=$(dirname "$filepath")

filename=$(basename "$filepath")
extension="${filename##*.}"
filename_without_extension="${filename%.*}"

new_filepath="$directory/${filename_without_extension}_image.$extension"

echo "Org filepath: $filepath" 
echo "New filepath: $new_filepath"



# convert to robomimic
python convert_robosuite.py --dataset $filepath 


# convert to image dataset
python dataset_states_to_obs.py --dataset  $filepath --output_name  $new_filepath --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# render videos
python hdf52videos.py --dataset  $new_filepath

