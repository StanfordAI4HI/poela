import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, num_actions)

        self.i1 = nn.Linear(state_dim, hidden_dim)
        self.i2 = nn.Linear(hidden_dim, hidden_dim)
        self.i3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class VAE_state(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim, max_state, device):
        super(VAE_state, self).__init__()
        self.e1 = nn.Linear(state_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, state_dim)

        self.max_state = max_state
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state):
        z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state.shape[0], z)

        return u, mean, std

    def decode(self, batch_size, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((batch_size, self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        s = F.relu(self.d1(z))
        s = F.relu(self.d2(s))
        if self.max_state is None:
            return self.d3(s)
        else:
            return self.max_state * torch.tanh(self.d3(s))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class discrete_PQL(object):
    def __init__(
            self,
            state_dim,
            num_actions,
            device,
            action_threshold=0.3,
            state_clipping=False,  # if true: PQL, false: BCQ
            log_pibs=True,
            density_estm="vae",
            beta=0.0,
            max_state=None,
            vmin = 0,
            hidden_dim=256,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
    ):
        self.log_pibs = log_pibs
        self.state_clipping = state_clipping

        self.device = device

        # Determine network type
        self.Q = FC_Q(state_dim, num_actions, hidden_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # VAE
        self.vmin = vmin

        self.density_estm = density_estm
        self.vae = VAE_state(state_dim, hidden_dim, state_dim * 2, torch.Tensor(max_state), device).to(device)
        self.best_vae = copy.deepcopy(self.vae)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        self.beta = beta

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = action_threshold

        # Number of training iterations
        self.iterations = 0

    def train_vae(self, replay_buffer, iterations):
        scores = []
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done, _, _ = replay_buffer.sample()

            recon, mean, std = self.vae(state)
            recon_loss = F.mse_loss(recon, state)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            scores.append(vae_loss.item())

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
        return np.mean(scores)

    def test_vae(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done, _, _ = replay_buffer.sample(batch_size)
        recon, mean, std = self.vae(next_state)
        recon_loss = ((recon - next_state) ** 2).mean(dim=1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        vae_loss = recon_loss + 0.5 * KL_loss
        return -vae_loss.detach().cpu().numpy()

    def save_best_vae(self):
        self.best_vae.load_state_dict(self.vae.state_dict())

    def load_best_vae(self):
        self.vae.load_state_dict(self.best_vae.state_dict())

    def save(self, filename):
        self.Q.save(filename+".pth")

    def load(self, filename):
        self.Q.load(filename+".pth")

    def select_action(self, state, pibs, nn_action_dist, eps=0):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q, imt, i = self.Q(state)
            if self.density_estm == "nn_action_dist":
                if isinstance(pibs, np.ndarray):
                    nn_action_dist = torch.FloatTensor(nn_action_dist).reshape(-1, self.num_actions).to(self.device)
                imt = (nn_action_dist <= self.threshold).float()
            elif self.log_pibs:
                assert pibs is not None
                if isinstance(pibs, np.ndarray):
                    pibs = torch.FloatTensor(pibs).reshape(-1, self.num_actions).to(self.device)
                imt = (pibs / pibs.max(1, keepdim=True)[0] > self.threshold).float()
            else:
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()

            # Use large negative number to mask actions from argmax
            greedy_a = int((imt * q + (1. - imt) * -1e8).argmax(1))

        if np.random.uniform(0, 1) > eps:
            return greedy_a
        else:
            return np.random.randint(self.num_actions)

    def get_prob(self, state, pibs, nn_action_dist, eps=0.01):
        a_id = self.select_action(state, pibs, nn_action_dist, eps)
        prob = np.zeros(self.num_actions)
        prob += eps * pibs
        prob[a_id] += (1.0 - eps)
        return prob

    def get_pib(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q, imt, i = self.Q(state)
            imt = imt.exp()
        return imt.cpu().numpy()

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done, pibs, dist_actions = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            if self.log_pibs:
                imt = (pibs / pibs.max(1, keepdim=True)[0] > self.threshold).float()
            else:
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()

            if self.density_estm == "nn_action_dist":
                imt = (dist_actions <= self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)

            if self.state_clipping:
                if self.density_estm == "vae":
                    recon, mean, std = self.vae(next_state)
                    recon_loss = ((recon - next_state) ** 2).mean(dim=1)
                    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
                    score = -recon_loss - 0.5 * KL_loss
                    score = score.reshape(replay_buffer.batch_size, -1).mean(dim=1, keepdim=True)
                    score = (score > self.beta).float()
                elif self.density_estm == "nn_action_dist":
                    score = (dist_actions.max(-1, keepdim=True) <= self.threshold).float()
                else:
                    raise NotImplementedError
            else:
                score = 1

            target_Q = reward + done * self.discount * score * q.gather(1, next_action).reshape(-1, 1) \
                       + done * self.discount * (1 - score) * self.vmin

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        if self.log_pibs:
            Q_loss = q_loss
        else:
            i_loss = F.nll_loss(imt, action.reshape(-1))
            Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
