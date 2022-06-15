#!/bin/bash
#SBATCH -c 1 --gres=gpu:0
#SBATCH --time=08:00:00

source ~/anaconda3/bin/activate base
cd /next/u/yaoliu/bcq_aiclinician

python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.0 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.0 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.0 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.0 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.0 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.25 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.25 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.25 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.25 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=0 --threshold=0.25 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=1 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=1 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=1 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=1 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=1 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=1 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=1 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=1 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=1 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=1 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=5 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=5 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=5 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=5 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.0 --beta_percentile=5 --seed=5  --max_timesteps=500

python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=5 --seed=1  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=5 --seed=2  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=5 --seed=3  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=5 --seed=4  --max_timesteps=500
python src/run_ql.py --env=$1 --state_clipping=1 --threshold=0.25 --beta_percentile=5 --seed=5  --max_timesteps=500