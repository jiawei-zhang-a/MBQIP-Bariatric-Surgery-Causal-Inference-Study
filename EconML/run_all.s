#!/bin/bash

# Define an array with the names of the subdirectories
folders=("CausalForestDML" "DML" "DMLOrthoForest" "DR" "Meta-Learners" "ORF")

# Loop over each subdirectory and submit the sbatch scripts
for folder in "${folders[@]}"
do
  # Check if run_BMI.s exists in the directory, if yes, then sbatch it
  if [[ -f "${folder}/run_BMI.s" ]]; then
    sbatch "${folder}/run_BMI.s"
  fi
  
  # Check if run_Logistic.s exists in the directory, if yes, then sbatch it
  if [[ -f "${folder}/run_Logistic.s" ]]; then
    sbatch "${folder}/run_Logistic.s"
  fi
done
