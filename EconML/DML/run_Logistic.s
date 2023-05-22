#!/bin/bash
#SBATCH --job-name=DML
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err

module purge;

cd ../..
source venv/bin/activate
export PATH=/scratch/jz4721/Observational-Study/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd EconML/DML
python CausalForest_Logistic.py $SLURM_ARRAY_TASK_ID

