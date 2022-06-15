import numpy as np
from sklearn.neighbors import NearestNeighbors

from envs.cancer import CancerSim, PolicyCancer


def offline_q_evaluation(q_policy, eval_buffer):
    states, _, _, _, _, pibs, nn_action_dist = eval_buffer.sample(eval_buffer.max_size)
    states = states[:,0,:]
    nn_action_dist = nn_action_dist[:, 0, :]
    pibs = pibs[:, 0, :]

    q, imt, i = q_policy.Q(states)
    if q_policy.density_estm == "nn_action_dist":
        imt = (nn_action_dist <= q_policy.threshold).float()
    elif q_policy.log_pibs:
        imt = (pibs / pibs.max(1, keepdim=True)[0] > q_policy.threshold).float()
    else:
        imt = imt.exp()
        imt = (imt / imt.max(1, keepdim=True)[0] > q_policy.threshold).float()

    # Use large negative number to mask actions from argmax
    values = (imt * q + (1. - imt) * q_policy.vmin).max(1)[0]
    return values.mean().cpu().item()


def offline_evaluation(q_policy, eval_buffer, weighted=True):
    states = eval_buffer.state
    actions = eval_buffer.action
    rewards = eval_buffer.reward[:, :, 0]
    estm_pibs = eval_buffer.estm_pibs
    pibs = eval_buffer.pibs
    not_dones = eval_buffer.not_done
    nn_action_dist = eval_buffer.nn_action_dist

    n = states.shape[0]
    horizon = states.shape[1]

    weights = np.ones((n, horizon))

    for idx in range(n):
        last = 1
        for t in range(horizon):
            pie = q_policy.get_prob(states[idx, t, :], estm_pibs[idx, t, :],
                                    nn_action_dist[idx, t, :], eps=0.01)
            a = actions[idx, t, 0]

            weights[idx, t] = last * (pie[a] / pibs[idx, t, a])
            last = weights[idx, t]

            if not not_dones[idx, t, 0]:
                weights[idx, t+1:] = weights[idx, t]
                break
    weights = np.clip(weights, 0, 1e3)
    if weighted:
        weights_norm = weights.sum(axis=0)
    else:
        weights_norm = weights.shape[0]
    weights /= weights_norm

    ess = (weights[:,-1].sum()) ** 2 / (((weights[:,-1]) ** 2).sum())
    estm = (weights[:,-1] * rewards.sum(axis=-1)).sum()

    return estm, ess


def online_evaluate(env, policy, trees, sample_size=1000):
    returns = []
    action_dict = dict()
    for i in range(sample_size):
        obs = env.reset()
        done = False
        rt = 0
        t = 0
        while not done:
            nn_action_dist = get_nn_action_dist(obs, trees)
            if isinstance(env, CancerSim):
                pibs = PolicyCancer().return_probs(obs, t)
            else:
                raise NotImplementedError
            a = policy.select_action(obs, pibs, nn_action_dist)
            t += 1
            obs, reward, done, _ = env.step(a)
            rt += reward
        returns.append(rt)
    return np.mean(returns)


def get_nn_action_dist(obs, trees):
    nn_action_dist = np.zeros(len(trees))
    for a in range(len(trees)):
        nn_action_dist[a] = (trees[a].kneighbors(obs.reshape(1,-1))[0].item()**2)/obs.size
    return nn_action_dist


def construct_nearest_neighbor_trees(buffer):
    num_actions = buffer.pibs.shape[-1]
    trees = dict()
    for a in range(num_actions):
        trees[a] = NearestNeighbors(n_neighbors=1)
        is_action = (np.squeeze(buffer.action,axis=-1) == a)
        trees[a].fit(buffer.state[is_action, :])
    return trees
