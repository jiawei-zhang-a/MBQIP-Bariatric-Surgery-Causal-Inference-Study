#!/bin/bash
#SBATCH --job-name=ORF_Logistic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --output=Logistic.out
#SBATCH --error=Logistic.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jz4721@nyu.edu

export OMP_NUM_THREADS=1

module purge

singularity exec --nv \
    --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/SCI/EconML/ORF/OrthoForest_Logistic.py"
