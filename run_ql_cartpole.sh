#!/usr/bin/env bash

source activate poela

python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.05 --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.05 --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.05 --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.05 --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.05 --seed=5  --max_timesteps=10000

python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.2 --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.2 --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.2 --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.2 --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0.2 --seed=5  --max_timesteps=10000
#
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0. --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0. --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0. --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0. --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=0 --threshold=0. --seed=5  --max_timesteps=10000


python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.05 --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.05 --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.05 --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.05 --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.05 --seed=5  --max_timesteps=10000


python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.2 --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.2 --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.2 --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.2 --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0.2 --seed=5  --max_timesteps=10000


python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0. --seed=1  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0. --seed=2  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0. --seed=3  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0. --seed=4  --max_timesteps=10000
python src/run_ql_cartpole.py --env=$1 --state_clipping=1 --threshold=0. --seed=5  --max_timesteps=10000
