#!/bin/bash
#SBATCH -c 1 --gres=gpu:0
#SBATCH --time=08:00:00

source ~/anaconda3/bin/activate base

cd /next/u/yaoliu/bcq_aiclinician
srun python src/run_ql.py --state_clipping=$1 --threshold=$2 --seed=1
srun python src/run_ql.py --state_clipping=$1 --threshold=$2 --seed=2
srun python src/run_ql.py --state_clipping=$1 --threshold=$2 --seed=3
srun python src/run_ql.py --state_clipping=$1 --threshold=$2 --seed=4
srun python src/run_ql.py --state_clipping=$1 --threshold=$2 --seed=5
