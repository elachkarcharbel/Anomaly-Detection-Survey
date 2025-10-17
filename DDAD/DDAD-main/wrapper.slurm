#!/usr/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name=DDAD_0s
#SBATCH --output=%x.%J.out  ## %x=job name, %J=job id
#SBATCH --error=%x.%J.out  ## %x=job name, %J=job id

# walltime (d-hh:mm::ss) max is 8 days
#SBATCH --time=5-00:00:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

## load modules
module load deep/pytorch-gpu/2.0.1


## execute your program
conda activate conda-env && python ddad_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/ddad
