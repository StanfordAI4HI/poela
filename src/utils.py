import pickle
import numpy as np
import torch


class EpisodicBuffer(object):
    def __init__(self, state_dim, num_actions, buffer_size, horizon, device):
        self.max_size = int(buffer_size)
        self.horizon = horizon
        self.device = device

        self.state = np.zeros((self.max_size, horizon, state_dim))
        self.action = np.zeros((self.max_size, horizon, 1))
        self.reward = np.zeros((self.max_size, horizon, 1))
        self.not_done = np.zeros((self.max_size, horizon, 1))
        self.pibs = np.zeros((self.max_size, horizon, num_actions))
        self.estm_pibs = np.zeros((self.max_size, horizon, num_actions))
        self.nn_action_dist = np.zeros((self.max_size, horizon, num_actions))

    def sample(self, batch_size):
        if batch_size > self.max_size:
            batch_size = self.max_size
        #ind = np.random.randint(0, self.max_size, size=batch_size)
        ind = np.random.choice(self.max_size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.pibs[ind]).to(self.device),
            torch.FloatTensor(self.estm_pibs[ind]).to(self.device),
            torch.FloatTensor(self.nn_action_dist[ind]).to(self.device)
        )

    def save(self, filename):
        data = {'observations': self.state,
                'actions': self.action,
                'rewards': self.reward,
                'not_done': self.not_done,
                'pibs': self.pibs,
                'estm_pibs': self.estm_pibs,
                'nn_action_dist': self.nn_action_dist,
                }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename, size=-1):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.state = data["observations"]
            self.action = data["actions"]
            self.reward = data["rewards"]
            self.not_done = data["not_done"]
            self.pibs = data["pibs"]
            if "estm_pibs" in data.keys():
                self.estm_pibs = data["estm_pibs"]
            else:
                self.estm_pibs = data["pibs"]
            self.nn_action_dist = data['nn_action_dist']
        self.max_size = self.state.shape[0]
        print(f"Episodic Buffer loaded with {self.max_size} episides.")


# Generic replay buffer for standard gym tasks
class SASRBuffer(object):
    def __init__(self, state_dim, num_actions, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        self.pibs = np.zeros((self.max_size, num_actions))  # p(a|s) under behavior policy, if we've logged them
        self.nn_action_dist = np.zeros(
            (self.max_size, num_actions))  # dist(s,s') where s' is the nearest neighbor of s with same action

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size > self.max_size:
            batch_size = self.max_size
        ind = np.random.randint(0, self.max_size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.pibs[ind]).to(self.device),
            torch.FloatTensor(self.nn_action_dist[ind]).to(self.device)
        )

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        n_episodes = data["observations"].shape[0]
        horizon = data["observations"].shape[1]
        n = 0
        for i in range(n_episodes):
            for h in range(horizon):
                self.state[n, :] = data["observations"][i, h, :]
                self.action[n, 0] = data["actions"][i, h, :]
                self.reward[n, 0] = data["rewards"][i, h, :]
                self.not_done[n, 0] = data["not_done"][i, h, :]
                if self.not_done[n, 0] and h < horizon:
                    self.next_state[n, :] = data["observations"][i, h+1, :]
                    if "estm_pibs" in data.keys():
                        self.pibs[n, :] = data["estm_pibs"][i, h + 1, :]
                    else:
                        self.pibs[n, :] = data["pibs"][i, h + 1, :]
                    self.nn_action_dist[n, :] = data['nn_action_dist'][i, h + 1, :]
                    n += 1
                else:
                    self.next_state[n, :] = data["observations"][i, h, :]
                    self.pibs[n, :] = data["pibs"][i, h, :]
                    self.nn_action_dist[n, :] = data['nn_action_dist'][i, h, :]
                    n += 1
                    break

        self.max_size = n
        print(f"Replay Buffer loaded with {self.max_size} transitions.")
