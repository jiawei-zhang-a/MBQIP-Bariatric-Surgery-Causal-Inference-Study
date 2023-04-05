#!/bin/bash
#SBATCH --job-name=CausalForest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=1:00:00
module purge;
module load anaconda3/2020.07;
cd ../../
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv;
export PATH=./penv/bin:$PATH;
python EconML/CausalForest/CausalForest_BMI.py