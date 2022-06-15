#!/usr/bin/env bash

source ~/anaconda3/bin/activate base
cd /next/u/yaoliu/bcq_aiclinician

# seed 1
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.0 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.0 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.0 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.0 --seed=1 --max_timesteps=500

# seed 2
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.0 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.0 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.0 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.0 --seed=2 --max_timesteps=500

# seed 3
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.0 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.0 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.0 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.0 --seed=3 --max_timesteps=500

# seed 4
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.0 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.0 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.0 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.0 --seed=4 --max_timesteps=500

# seed 5
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.0 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.0 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.0 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.0 --seed=5 --max_timesteps=500

# seed 1
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.1 --seed=1 --max_timesteps=500

# seed 2
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.1 --seed=2 --max_timesteps=500

# seed 3
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.1 --seed=3 --max_timesteps=500

# seed 4
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.1 --seed=4 --max_timesteps=500

# seed 5
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=0.1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=0.1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=0.1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=0.1 --seed=5 --max_timesteps=500


# seed 1
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=1 --seed=1 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=1 --seed=1 --max_timesteps=500

# seed 2
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=1 --seed=2 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=1 --seed=2 --max_timesteps=500

# seed 3
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=1 --seed=3 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=1 --seed=3 --max_timesteps=500

# seed 4
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=1 --seed=4 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=1 --seed=4 --max_timesteps=500

# seed 5
python src/run_pg.py --env=$1 --action_mask_type=step --threshold=0.0 --var_coeff=1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.5 --var_coeff=1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.1 --var_coeff=1 --seed=5 --max_timesteps=500
python src/run_pg.py --env=$1 --action_mask_type=nn_action_dist --threshold=0.05 --var_coeff=1 --seed=5 --max_timesteps=500
