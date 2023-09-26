#!/bin/bash
#SBATCH --job-name=CausalDML_Logistic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=70:00:00
#SBATCH --output=BMI.out
#SBATCH --error=BMI.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jz4721@nyu.edu

module purge;

cd ../..
source venv/bin/activate
export PATH=/scratch/jz4721/Observational-Study/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd EconML/DML
python CausalForest_BMI.py