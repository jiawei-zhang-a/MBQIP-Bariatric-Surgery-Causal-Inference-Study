#!/bin/bash
#SBATCH --job-name=riskCausalForest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=riskCausalForest.out
#SBATCH --error=riskCausalForest.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jz4721@nyu.edu

export OMP_NUM_THREADS=1

module purge

singularity exec --nv \
    --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/SCI/EconML/CausalForestDML/riskCausalForest.py"
