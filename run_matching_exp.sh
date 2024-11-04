#!/bin/bash

# Array of months for the first descriptor
#months1=("may" "june" "september")

# Array of months for the second descriptor
months2=("march" "april" "may" "june" "september")

# Descriptors
#descriptor1="64N-192U-FN"
descriptor2="U-64U-196U-FN-SPBG"

# Loop over each month and run the command for the first descriptor
#for month in "${months1[@]}"
#do
    #echo "Running match_pairs.py for month: $month with descriptor: $descriptor1"
    #./match_pairs.py --eval --viz --month "$month" --desc "$descriptor1"
#done

# Loop over each month and run the command for the second descriptor
for month in "${months2[@]}"
do
    echo "Running match_pairs.py for month: $month with descriptor: $descriptor2"
    ./match_pairs.py --eval --viz --cache --month "$month" --desc "$descriptor2"
done

echo "All tasks completed."

