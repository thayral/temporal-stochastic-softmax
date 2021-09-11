#!/bin/bash
#SBATCH --time=22:50:00
#SBATCH --account=rrg-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=16G               # memory (per node)
# set name of job
#SBATCH --job-name=thayral_pytorch
#SBATCH --output=%x-%j.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=theo.ayral.1@etsmtl.net


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export DELI_WD=$(pwd)


nvidia-smi

echo $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/work

cd $SLURM_TMPDIR


tar -xf ~/projects/def-mpederso/thayral/AFEW/FebSeetaCompleted.tar.gz -C $SLURM_TMPDIR


module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r ~/reqsight3.txt


python $DELI_WD/main.py
