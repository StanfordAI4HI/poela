#!/bin/bash
#SBATCH -c 1 --gres=gpu:0
#SBATCH --time=08:00:00

source ~/anaconda3/bin/activate base

cd /next/u/yaoliu/bcq_aiclinician
srun python src/run_pg.py --action_mask_type=$1 --am_coeff=$2 --threshold=$3 --var_coeff=$4 --seed=1
srun python src/run_pg.py --action_mask_type=$1 --am_coeff=$2 --threshold=$3 --var_coeff=$4 --seed=2
srun python src/run_pg.py --action_mask_type=$1 --am_coeff=$2 --threshold=$3 --var_coeff=$4 --seed=3
srun python src/run_pg.py --action_mask_type=$1 --am_coeff=$2 --threshold=$3 --var_coeff=$4 --seed=4
srun python src/run_pg.py --action_mask_type=$1 --am_coeff=$2 --threshold=$3 --var_coeff=$4 --seed=5