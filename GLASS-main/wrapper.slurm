#!/usr/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name=GLASS_0s
#SBATCH --output=%x.%J.out  ## %x=job name, %J=job id
#SBATCH --error=%x.%J.out  ## %x=job name, %J=job id

# walltime (d-hh:mm::ss) max is 8 days
#SBATCH --time=5-00:00:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

## load modules
module load deep/pytorch-gpu/2.0.1


## execute your program
conda activate conda-env2 && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && export TORCH_HOME=/Work/Users/celachkar/GLASS/.torch_cache && python glass_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/glass
