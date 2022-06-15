#!/bin/bash

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 1 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 1 0.01
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 1 0.05
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 1 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 1 0.3

sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 0 0.0
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 0 0.01
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 0 0.05
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 0 0.1
sbatch -p next -c 2 --time=48:00:00 --mem=4096 run_ql_mimic.sh 0 0.3