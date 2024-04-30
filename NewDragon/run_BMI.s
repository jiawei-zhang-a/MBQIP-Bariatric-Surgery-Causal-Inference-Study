#!/bin/bash
#SBATCH --job-name=DNN-BMI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=79:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jz4721@nyu.edu
#SBATCH --output=BMI.out
#SBATCH --error=BMI.err

export OMP_NUM_THREADS=1

module purge

singularity exec --nv \
    --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/SCI/NewDragon/dragonnet_BMI.py"
