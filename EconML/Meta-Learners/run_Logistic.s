#!/bin/bash
#SBATCH --job-name=Meta-Learners_Logistic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=29:00:00
#SBATCH --output=Logistic.out
#SBATCH --error=Logistic.err

module purge;

cd ../..
source venv/bin/activate
export PATH=/scratch/jz4721/Observational-Study/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd EconML/Meta-Learners
python Meta-Learners_Logistic.py