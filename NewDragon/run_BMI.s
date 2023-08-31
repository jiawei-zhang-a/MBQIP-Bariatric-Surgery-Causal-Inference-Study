#!/bin/bash
#SBATCH --job-name=DNN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jz4721@nyu.edu
#SBATCH --output=BMI.out
#SBATCH --error=BMI.err
#SBATCH --gres=gpu:1

module purge;

cd ..
source venv/bin/activate
export PATH=/scratch/jz4721/SCI/venv/lib64/python3.8/bin:$PATH

cd NewDragon/
python dragonnet_BMI.py