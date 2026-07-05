#!/bin/bash

prefix="PPBM"

PARAM_FILE="parameters.txt"
SLURM_SCRIPT="script.sh"

while read -r -a params; do
    
    OLD_IFS="$IFS"
    IFS="_"
    JOB_ID="${prefix}_${params[*]}"
    IFS="$OLD_IFS"
    
    # Submit the job and pass the entire array of parameters
    sbatch --job-name="${JOB_ID}" \
           --output="output/${JOB_ID}.out" \
           --error="error/${JOB_ID}.err" \
           "$SLURM_SCRIPT" "${params[@]}"

    # print that the job has been submitted
    echo "Parameters: ${params[@]}"
done < "$PARAM_FILE"