from cancer import CancerSim, PolicyCancer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import tqdm
import matplotlib.pyplot as plt

from collections import deque


class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, state):
        return self.output(state)


class ReplayBuffer(object):
    def __init__(self, state_dim, device="cpu", max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, 1), int)
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def online_Q(env, n=5000):
    model = FC_Q(env.state_dim, env.num_actions)
    target_model = FC_Q(env.state_dim, env.num_actions)
    optimizer = torch.optim.Adam(model.parameters())
    buffer = ReplayBuffer(env.state_dim)
    rewards = deque(maxlen=100)
    eps = 1.0
    for i in range(n):
        obs = env.reset()
        reward = 0
        done = False
        while not done:
            q = model.forward(torch.from_numpy(obs).float())

            if np.random.rand() < eps:
                action = np.random.choice(env.num_actions)
            else:
                action = q.max(-1)[1]

            next_obs, rt, done, _ = env.step(action)
            reward += rt
            buffer.add(obs, action, next_obs, rt, done)
            obs = next_obs
        rewards.append(reward)
        eps *= 0.995

        if i%100 == 0:
            print(f"Episode {i}: mean rewards {round(np.mean(rewards), 2)}")
            target_model.load_state_dict(model.state_dict())

        if buffer.size < 100:
            continue
        state, action, next_state, reward, not_done = buffer.sample(batch_size=100)

        state_action_values = model(state).gather(1, action)
        target_Q = target_model(next_state).max(1)[0]
        target_values = reward.reshape(-1, 1) + not_done.reshape(-1, 1) * 0.99 * target_Q.reshape(-1, 1)
        target_values = target_values.detach()
        loss = torch.nn.MSELoss()(state_action_values, target_values)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    obs = env.reset()
    r = 0
    done = False
    print(obs[1:4].sum())
    while not done:
        q = model.forward(torch.from_numpy(obs).float())
        action = q.max(-1)[1].item()
        # print(q)
        obs, rt, done, _ = env.step(action)
        print("action:", action, "State:", obs, "reward:", rt)
        r += rt
    print(obs[1])
    print(r)


def test_fix_policy(env, eps=0, mon=9, n=1000):
    policy = PolicyCancer(eps_behavior=eps, months_for_treatment=mon)
    rewards = []
    for i in range(n):
        obs = env.reset()
        t = 0
        reward = 0
        done = False
        while not done:
            # print(obs)
            # a = np.random.choice([0,1], p=[eps, 1-eps])
            a = policy(obs, t)
            obs, rt, done, _ = env.step(a)
            t += 1
            reward += rt
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards), rewards


def collect_data(n, type="PCV", penalty=0.1):
    episodes_per_policy = 100
    env = CancerSim(dose_penalty=penalty, therapy_type=type, env_seed=None, max_steps=30, transition_noise=0.0)

    dataset = {'observations': np.zeros((n, env.max_steps, env.state_dim)),
               'actions': np.zeros((n, env.max_steps, 1), np.int),
               'rewards': np.zeros((n, env.max_steps, 1)),
               'not_done': np.zeros((n, env.max_steps, 1)),
               'pibs': np.zeros((n, env.max_steps, env.num_actions)),
               'nn_action_dist': np.ones((n, env.max_steps, env.num_actions)) * 1e9,
               }

    for i in range(n):
        if (i % episodes_per_policy) == 0:
            policy = PolicyCancer()

        env = CancerSim(dose_penalty=penalty, therapy_type=type, env_seed=None, max_steps=30, transition_noise=0.0)
        obs = env.reset()
        t = 0
        done = False
        while not done:
            a = policy(obs, t)
            new_obs, rt, done, _ = env.step(a)

            dataset['observations'][i, t, :] = obs
            dataset['actions'][i, t, :] = a
            dataset['rewards'][i, t, :] = rt
            dataset['not_done'][i, t, :] = float(1-done)
            dataset['pibs'][i, t, :] = policy.return_probs(obs, t)

            obs = new_obs
            t += 1

    X = dataset['observations'].reshape((-1, env.state_dim))
    A0 = (dataset['actions'].reshape((-1, env.state_dim)) != 0)
    A1 = (dataset['actions'].reshape((-1, env.state_dim)) != 1)
    for i in tqdm.tqdm(range(n)):
        for t in range(env.max_steps):
            dist = (dataset['observations'][i, t, :] - X)**2
            dist = dist.mean(axis=1)

            a0_dist = np.ma.masked_array(dist, A0)
            a1_dist = np.ma.masked_array(dist, A1)

            #print(dist, a0_dist.mean())

            dataset['nn_action_dist'][i, t, :] = np.array([a0_dist.min(), a1_dist.min()])

    return dataset


if __name__ == '__main__':

    env = CancerSim(dose_penalty=1.0, therapy_type="PCV", a_bins=1, env_seed=None, max_steps=30)
    online_Q(env, 5000)

    for i in range(10):
        m, std, rewards = test_fix_policy(env, eps=0, mon=i)
        print("treat months:", i, "mean reward:", m)
