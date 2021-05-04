#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH -o "/scratch/inf0/user/gtiwari/slurm-%A.out"
#SBATCH -e "/scratch/inf0/user/gtiwari/slurm-%A.err"
#SBATCH --gres gpu:1

echo "NASA pytorch implementation generation"
cd /BS/garvita/work/code/NASA_pytorch
source /BS/garvita/static00/software/miniconda3/etc/profile.d/conda.sh
conda activate if-net_10

python generator.py --models D -checkpoint 763 -mw 0.0

