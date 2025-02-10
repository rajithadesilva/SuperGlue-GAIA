#!/bin/bash

# Array of months for the first descriptor
#months1=("may" "june" "september")

# Array of months for the second descriptor
months=("march" "april" "may" "june" "september")
path="test"

# Loop over each month and run the command for the first descriptor
#for month in "${months1[@]}"
#do
    #echo "Running match_pairs.py for month: $month with descriptor: $descriptor1"
    #./match_pairs.py --eval --viz --month "$month" --desc "$descriptor1"
#done

# Loop over each month and run the command for the second descriptor
for month in "${months[@]}"
do
    #echo "Running evo APE for month: $month"
    #evo_ape tum "$path"/"$month"/gt_poses_"$month".tum "$path"/"$month"/recovered_poses_"$month".tum  --align --correct_scale -va --plot
    echo "Running evo RPE for month: $month"
    evo_rpe tum "$path"/gt_poses_"$month".tum "$path"/recovered_poses_"$month".tum  --align --correct_scale --plot
done

echo "All tasks completed."

