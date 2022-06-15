#!/bin/bash

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.4 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.6 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.8 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 1.0 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 2.0 0.0

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.4 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.6 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.8 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 1.0 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 2.0 0.1

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.4 1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.6 1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.8 1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 1.0 1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 2.0 1

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.4 10
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.6 10
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 0.8 10
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 1.0 10
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run.sh nn_action_dist 0 2.0 10
#
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.01 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.02 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.03 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.05 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.1 0.0

sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.01 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.02 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.03 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.05 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.1 0.1

sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0 1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.01 1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.02 1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.03 1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.05 1
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.1 1

sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0 10
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.01 10
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.02 10
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.03 10
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.05 10
sbatch -p next -c 2 --time=48:00:00 --mem=8192 run.sh step 0 0.1 10