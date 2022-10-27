import argparse
import numpy as np
import torch
import os

from envs.cancer import CancerSim
from offline_pg import ISPG
from evaluation import online_evaluate, construct_nearest_neighbor_trees
import utils

try:
    import dill as pickle
except ImportError:
    import pickle


def train_PG(state_dim, num_actions, device, args, parameters):
    # Load training and validation data
    train_buffer = utils.EpisodicBuffer(state_dim, num_actions, parameters["buffer_size"], parameters["horizon"],
                                        device)
    train_buffer.load(parameters["buffer_name"]+"_train_episodes")
    validation_buffer = utils.EpisodicBuffer(state_dim, num_actions, parameters["buffer_size"], parameters["horizon"],
                                             device)
    validation_buffer.load(parameters["buffer_name"]+"_val_episodes")

    if parameters["self_normalized"]:
        parameters["buffer_size"] = train_buffer.max_size
        parameters["batch_size"] = train_buffer.max_size

    # Initialize and load policy
    policy = ISPG(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        horizon=parameters["horizon"],
        var_coeff=parameters["var_coeff"],
        discount=parameters["discount"],
        step_average=parameters["step_average"],
        optimizer=parameters["optimizer"],
        optimizer_parameters=parameters["optimizer_parameters"],
        threshold=parameters["threshold"],
        using_cv=parameters["using_cv"],
        cv_type=parameters["cv_type"],
        traj_clipping=parameters["traj_clipping"],
        action_mask_type=parameters["action_mask_type"],
        self_normalized=parameters["self_normalized"],
        train_sample_size=train_buffer.max_size,
        hidden_dim=parameters["hidden_dim"]
    )
    if parameters["using_cv"]:
        policy.set_cv(train_buffer)

    training_iters = 0
    while training_iters < parameters["bc_steps"]:
        for _ in range(100):
            policy.behavior_cloning(train_buffer, parameters["bc_batch_size"])
        training_iters += 100
        print(f"Training iterations: {training_iters}")
        policy.test_var(train_buffer)
        policy.test_var(validation_buffer, is_validation=True)
        policy.test_nll(train_buffer)
        policy.test_nll(validation_buffer,is_validation=True)
    policy.reset_optimizer()

    val_w_scores = []
    train_w_scores = []
    training_iters = 0
    best_wis = -np.inf
    best_noess_wis = -np.inf

    test_scores = []
    best_eval = -np.inf
    knn_trees = None
    while training_iters < args.max_timesteps:
        for _ in range(int(parameters["eval_freq"])):
            policy.train(train_buffer, parameters["batch_size"])

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")

        train_w_scores.append(policy.test_var(train_buffer))
        np.save(f"./results/{parameters['save_f_name']}_train_wis_scores", np.array(train_w_scores))
        val_w_scores.append(policy.test_var(validation_buffer, is_validation=True))
        np.save(f"./results/{parameters['save_f_name']}_val_wis_scores", np.array(val_w_scores))

        if val_w_scores[-1][0] > best_wis and (val_w_scores[-1][-1] > 200 or args.env != "mimic_sepsis"):
            policy.save(f"./models/{parameters['save_f_name']}_best_")
            print("New best score on validation set", val_w_scores[-1][0])
            best_wis = val_w_scores[-1][0]
        elif val_w_scores[-1][0] > best_noess_wis and args.env == "mimic_sepsis":
            policy.save(f"./models/{parameters['save_f_name']}_best_noess_")
            print("New best score (no ess constraint) on validation set", val_w_scores[-1][0])
            best_noess_wis = val_w_scores[-1][0]

        policy.save(f"./models/{parameters['save_f_name']}_")

        if "cancer_mdp_pcv" in args.env:
            if knn_trees is None:
                knn_trees = construct_nearest_neighbor_trees(train_buffer)
            env = CancerSim(dose_penalty=1.0, therapy_type="PCV", env_seed=None, max_steps=30, state_dim=5)
            eval = online_evaluate(env, policy, knn_trees, 20)
            test_scores.append(eval)
            np.save(f"./results/{parameters['save_f_name']}_true_scores", np.array(test_scores))
            print(f"Test score: {eval:.5e}")
            if eval > best_eval:
                policy.save(f"./models/{parameters['save_f_name']}_test_best_")
                print("New best test score", eval)
                best_eval = eval


if __name__ == "__main__":

    mimic_parameters = {
        # Learning
        "horizon": 20,
        "discount": 1,
        "buffer_size": 8982, #8982, 7485
        "batch_size": 8982,
        "bc_batch_size": 100,
        "hidden_dim": 256,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "bc_steps": 500,
        "eval_freq": 10,
        "var_coeff": 0.01,
        "step_average": False,
        "threshold": 0.0001,
        "using_cv": False,
        "cv_type": "const_mean", # const, const_mean, behavior_q, target_q
        "traj_clipping": True,
        "action_mask_type": "nn_action_dist", # step, trajectory, nn_action_dist
        "self_normalized": True,

        "save_f_name": "PG_mimic45"
    }

    cancer_parameters = {
        # Learning
        "horizon": 30,
        "discount": 1,
        "buffer_size": 10000,
        "batch_size": 10000,
        "bc_batch_size": 100,
        "hidden_dim": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "bc_steps": 500,
        "eval_freq": 10,
        "var_coeff": 0.01,
        "step_average": False,
        "threshold": 0.2,
        "using_cv": False,
        "cv_type": "const_mean",  # const, const_mean, behavior_q, target_q
        "traj_clipping": True,
        "action_mask_type": "nn_action_dist",  # step, trajectory, nn_action_dist
        "self_normalized": True,

        "save_f_name": "PG_mimic45"
    }

    cartpole_parameters = {
        # Learning
        "horizon": 200,
        "discount": 1,
        "buffer_size": 744,
        "batch_size": 744,
        "bc_batch_size": 100,
        "hidden_dim": 256,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "bc_steps": 500,
        "eval_freq": 100,
        "var_coeff": 0.01,
        "step_average": False,
        "threshold": 0.0001,
        "using_cv": False,
        "cv_type": "const_mean",  # const, const_mean, behavior_q, target_q
        "traj_clipping": True,
        "action_mask_type": "nn_action_dist",  # step, trajectory, nn_action_dist
        "self_normalized": True,

        "save_f_name": "PG_cartpole"
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="mimic_sepsis")
    parser.add_argument("--seed", default=-1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1000, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--var_coeff", default=0.01, type=float)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--action_mask_type", default="nn_action_dist", type=str)
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(low=1000000)

    print("---------------------------------------")
    print(f"Setting: Training PG, Env: {args.env}, Seed: {args.seed}")
    print(args)
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Make env and determine properties
    if args.env == "mimic_sepsis":
        state_dim = 45
        num_actions = 25
        parameters = mimic_parameters

        experiment_name = "PG_mimic45"
        parameters["buffer_name"] = "./data/s45da_mimic"
    elif "cancer_pcv" in args.env:
        state_dim = 3
        num_actions = 2
        parameters = cancer_parameters

        experiment_name = "PG_" + args.env
        parameters["buffer_name"] = "./data/" + args.env
    elif "cancer_mdp_pcv" in args.env:
        state_dim = 5
        num_actions = 2
        parameters = cancer_parameters

        experiment_name = "PG_" + args.env
        parameters["buffer_name"] = "./data/" + args.env
    elif args.env == "cartpole":
        state_dim = 3 #todo 4
        num_actions = 2
        parameters = cartpole_parameters

        experiment_name = "2102/PG_cartpole"
        parameters["buffer_name"] = "./data/cartpole"
    else:
        raise NotImplementedError

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parameters["threshold"] = args.threshold
    parameters["var_coeff"] = args.var_coeff
    parameters["action_mask_type"] = args.action_mask_type

    settings = f"{experiment_name}_mask{parameters['action_mask_type']}_" \
               f"thresh{args.threshold}var{args.var_coeff}seed{args.seed}"
    parameters["save_f_name"] = settings

    print("\nParameters:")
    print(parameters)
    train_PG(state_dim, num_actions, device, args, parameters)
