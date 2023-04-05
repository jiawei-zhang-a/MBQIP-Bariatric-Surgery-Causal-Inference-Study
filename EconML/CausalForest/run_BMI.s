#!/bin/bash
#SBATCH --job-name=CausalForest_Super
#SBATCH --nodes=1
#SBATCH --cpus-per-task=99
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3:00:00
module purge;

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.9/bin:$PATH
source ~/.bashrc

python3 CausalForest_BMI.py