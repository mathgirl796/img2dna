#!/bin/bash
#SBATCH -p nonfs
#SBATCH -J mengma
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --chdir=/home/user/duanran/repo/mengma/

. ~/.bashrc
conda activate mengma

python run_real_test.py
