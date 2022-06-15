import argparse

import numpy as np
import torch

from offline_ql import discrete_PQL
from utils import SASRBuffer, EpisodicBuffer
from evaluation import online_evaluate, offline_evaluation, offline_q_evaluation, construct_nearest_neighbor_trees
from envs.cancer import CancerSim

try:
    import dill as pickle
except ImportError:
    import pickle
import os


# Trains BCQ offline
def train_Qlearning(num_actions, state_dim, device, args, parameters):

    # Initialize and load policy
    policy = discrete_PQL(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        action_threshold=args.threshold,
        state_clipping=parameters["state_clipping"],
        log_pibs=parameters["log_pibs"],
        max_state=parameters["max_state"],
        vmin=parameters["vmin"],
        hidden_dim=parameters["hidden_dim"],
        discount=parameters["discount"],
        optimizer=parameters["optimizer"],
        optimizer_parameters=parameters["optimizer_parameters"],
        polyak_target_update=parameters["polyak_target_update"],
        target_update_frequency=parameters["target_update_freq"],
        tau=parameters["tau"],
    )

    training_buffer = SASRBuffer(state_dim, num_actions, parameters["batch_size"], parameters["buffer_size"], device)
    training_buffer.load(parameters["buffer_name"]+"_train_episodes")
    valid_buffer = SASRBuffer(state_dim, num_actions, parameters["batch_size"], parameters["buffer_size"], device)
    valid_buffer.load(parameters["buffer_name"] + "_val_episodes")
    valid_episodes = EpisodicBuffer(state_dim, num_actions, parameters["buffer_size"], parameters["horizon"], device)
    valid_episodes.load(parameters["buffer_name"]+"_val_episodes")

    if parameters["state_clipping"]:
        if parameters["density_estm"] == "vae":
            training_iters = 0
            best_loss = np.inf
            while training_iters < int(10000):
                vae_loss = policy.train_vae(training_buffer, iterations=1000)
                test_loss = policy.test_vae(valid_buffer, batch_size=10000)
                test_loss = -np.mean(test_loss)
                training_iters += 1000
                print(
                    f"Training iterations: {training_iters}. VAE train loss: {vae_loss:.3f}. test loss: {test_loss:.3f}")
                if test_loss < best_loss:
                    best_loss = test_loss
                    policy.save_best_vae()
            policy.load_best_vae()
            density_scores = policy.test_vae(training_buffer, batch_size=training_buffer.max_size)

            beta = np.percentile(density_scores, parameters["beta_percentile"])
            policy.beta = beta
            print(parameters["beta_percentile"], " percentile:", beta)

    training_iters = 0
    val_scores = []
    train_scores = []
    best_val_eval = -np.inf
    best_noess_val_eval = -np.inf

    test_scores = []
    best_eval = -np.inf
    knn_trees = None
    while training_iters < args.max_timesteps:

        for _ in range(int(parameters["eval_freq"])):
            policy.train(training_buffer)
        training_iters += int(parameters["eval_freq"])

        print(f"Training iterations: {training_iters}")
        qvalues = offline_q_evaluation(policy, valid_episodes)
        valid_wis, valid_ess = offline_evaluation(policy, valid_episodes, weighted=True)
        train_scores.append(qvalues)
        val_scores.append([valid_wis,valid_ess])
        np.save(f"./results/{parameters['save_f_name']}_q_scores", np.array(train_scores))
        np.save(f"./results/{parameters['save_f_name']}_val_scores", np.array(val_scores))

        print(f"Average Q value: {qvalues:.5e} Validation WIS: {valid_wis:.5e}, ESS: {valid_ess:.5e}")

        if valid_wis > best_val_eval and (valid_ess > 200 or args.env != "mimic_sepsis"):
            policy.save(f"./models/{parameters['save_f_name']}_best_")
            print("New best score on validation set", valid_wis)
            best_val_eval = valid_wis
        elif valid_wis > best_noess_val_eval and (args.env == "mimic_sepsis"):
            policy.save(f"./models/{parameters['save_f_name']}_best_noess_")
            print("New best score (no ess constraint) on validation set", valid_wis)
            best_noess_val_eval = valid_wis

        policy.save(f"./models/{parameters['save_f_name']}")

        if "cancer_mdp_pcv" in args.env:
            if knn_trees is None:
                knn_trees = construct_nearest_neighbor_trees(training_buffer)
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
        "hidden_dim": 256,
        "eval_freq": 100,
        "discount": 0.99,
        "buffer_size": 200000,
        "batch_size": 100,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005,

        "state_clipping": False,
        "density_estm": "vae",  # vae, nn_action_dist
        "log_pibs": False,
        "beta_percentile": 2.0,  # clip how many samples with vae

        # Domain parameter
        "horizon": 20,
        "max_state": [20, 0.5000, 0.5000, 5.1672, 2.7543, 7.9158, 2.7949, 5.2763, 7.1967,
                      9.0866, 11.8737, 7.8319, 236.7704, 3.0286, 9.9485, 8.7237, 6.2627, 15.6281,
                      8.5191, 13.5491, 6.1870, 40.5525, 10.0892, 6.7538, 20.4172, 8.5691, 6.5705,
                      12.9162, 6.6171, 17.8618, 29.6399, 5.0306, 11.4043, 40.8348, 5.0392, 2.3167,
                      23.0958, 7.7701, 7.5767, 5.6300, 4.9695, 4.2235, 1.7962, 2.8099, 2.0322],
        "vmin": 0,
    }

    cancer_parameters = {
        "hidden_dim": 32,
        "eval_freq": 100,
        "discount": 0.99,
        "buffer_size": 150000,
        "batch_size": 100,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005,

        "state_clipping": False,
        "density_estm": "vae",  # vae, nn_action_dist
        "log_pibs": True,
        "beta_percentile": 2.0,  # clip how many samples with vae
        # Domain parameter
        "horizon": 30,
        "max_state": [3.25401319, 51.84731224, 30.],
        "vmin": -30,
    }

    cancer_mdp_parameters = {
        "hidden_dim": 32,
        "eval_freq": 10,
        "discount": 0.99,
        "buffer_size": 300000,
        "batch_size": 100,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005,

        "state_clipping": False,
        "density_estm": "vae",  # vae, nn_action_dist
        "log_pibs": True,
        "beta_percentile": 2.0,  # clip how many samples with vae
        # Domain parameter
        "horizon": 30,
        "max_state": [3.18970173,  7.85591301, 44.83595683, 30.64735325, 30.],
        "vmin": -30,
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="mimic_sepsis")
    parser.add_argument("--seed", default=-1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=10000, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--state_clipping", default=1, type=int)
    parser.add_argument("--threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--beta_percentile", default=2.0, type=float)  # Threshold hyper-parameter for BCQ
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(low=1000000)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if args.state_clipping == 1:
        experiment_name = "PQL_"
    else:
        experiment_name = "BCQ_"

    print("---------------------------------------")
    print(f"Setting: Training {experiment_name}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Make env and determine properties
    if args.env == "mimic_sepsis":
        state_dim = 45
        num_actions = 25
        parameters = mimic_parameters

        experiment_name += "mimic45"
        parameters["buffer_name"] = "./data/s45da_mimic"
    elif "cancer_pcv" in args.env:
        state_dim = 3
        num_actions = 2
        parameters = cancer_parameters

        experiment_name += args.env
        parameters["buffer_name"] = "./data/" + args.env
    elif "cancer_mdp_pcv" in args.env:
        state_dim = 5
        num_actions = 2
        parameters = cancer_mdp_parameters

        experiment_name += args.env
        parameters["buffer_name"] = "./data/" + args.env
    else:
        raise NotImplementedError

    parameters["state_clipping"] = (args.state_clipping==1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    parameters["save_f_name"] = f"{experiment_name}_thresh{args.threshold}_beta{args.beta_percentile}_seed{args.seed}"
    parameters["beta_percentile"] = args.beta_percentile

    print(parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_Qlearning(num_actions, state_dim, device, args, parameters)
