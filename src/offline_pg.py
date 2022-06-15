import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Policy(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super(MLP_Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        p = F.relu(self.l1(state))
        p = F.relu(self.l2(p))
        p = F.relu(self.l3(p))
        return F.log_softmax(p, dim=-1)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class ISPG(object):

    def __init__(self,
                 state_dim,
                 num_actions,
                 device,
                 horizon = 20,
                 var_coeff=0.001,
                 discount=1.0,
                 step_average=False,
                 self_normalized=False,
                 optimizer="Adam",
                 optimizer_parameters={},
                 threshold = 0.03,
                 using_cv = False,
                 cv_type = "const",
                 traj_clipping = False,
                 action_mask_type = "step", # step, nn_action_dist
                 train_sample_size = -1,
                 hidden_dim=256,
                 ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device

        self.horizon = horizon
        self.discount = discount
        self.var_coeff = var_coeff
        self.step_average = step_average
        self.threshold = threshold
        self.traj_clipping = traj_clipping
        self.action_mask_type = action_mask_type
        if self.action_mask_type == "trajectory":
            self.traj_prob_threshold = np.zeros(self.horizon)

        self.policy = MLP_Policy(state_dim, num_actions, hidden_dim).to(device)
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.policy_optimizer = getattr(torch.optim, optimizer)(self.policy.parameters(), **optimizer_parameters)

        self.train_sample_size = train_sample_size
        self.self_normalized = self_normalized
        self.using_cv = using_cv
        self.cv_type = cv_type
        self.mean_return = None
        self.A = 0
        self.B = 0
        self.iteration = 0

    def pg_loss(self, states, actions, rewards, not_dones, pibs, nn_action_dist):
        n = states.shape[0]

        if self.self_normalized:
            assert n == self.train_sample_size
            clis, ESS, is_weights = self.compute_dr(states, actions, rewards, not_dones, pibs, nn_action_dist, weighted=True)
            loss = - clis.mean() + self.var_coeff * ((clis - clis.mean() * is_weights) ** 2).sum().sqrt()/n

        else:
            clis, ESS, is_weights = self.compute_dr(states, actions, rewards, not_dones, pibs, nn_action_dist)
            loss = - clis.mean() + (self.var_coeff/np.sqrt(n)) * (self.B * (clis ** 2).sum() / (n - 1) - self.A * estm.sum() / (n - 1))
        return loss

    def bc_loss(self, states, actions, not_dones, pibs):
        log_prob = not_dones*((self.policy(states)*pibs).sum(dim=-1,keepdim=True))
        return -log_prob.mean()

    def discounted_return(self, rewards):
        returns = torch.zeros_like(rewards[:,0])
        for t in range(self.horizon):
            returns += pow(self.discount, t)*rewards[:,t]
        return returns

    def compute_dr(self, states, actions, rewards, not_dones, pibs, nn_action_dist,
                   estm_pibs=None, weighted=False, eval_mode=False):
        if estm_pibs is None:
            estm_pibs = pibs
        log_probs_all = self.policy(states)
        behaviors = pibs.gather(-1, actions)
        valid_step = torch.cat((torch.ones_like(not_dones[:,0:1]),
                                not_dones[:,:self.horizon-1]), dim=1)

        if self.action_mask_type == "step":
            mask = (estm_pibs >= self.threshold).float()
        elif self.action_mask_type == "nn_action_dist":
            mask = (nn_action_dist <= self.threshold).float()
        else:
            raise NotImplementedError
        probs_all = log_probs_all.exp() * (mask + (mask.sum(dim=-1, keepdim=True) == 0).float())
        probs_all = probs_all / probs_all.sum(dim=-1, keepdim=True).clamp(1e-20,1)
        probs = probs_all.gather(-1, actions)

        is_weights = torch.ones_like(probs[:, 0, :])
        for t in range(self.horizon):
            if t > 0:
                assert (valid_step[:, t] != not_dones[:,t-1]).sum().item() == 0
            filtered_probs_t = valid_step[:, t] * probs[:, t].clone() + (-valid_step[:, t]+1)
            is_weights = is_weights * (filtered_probs_t / behaviors[:, t, :])

        is_weights = torch.clamp(is_weights, 0, 1e3)
        if weighted:
            is_weights = is_weights / (is_weights.mean())
            clis = is_weights*self.discounted_return(rewards)#rewards.sum(dim=1)
        else:
            if self.using_cv:
                clis = is_weights * (self.discounted_return(rewards)-self.mean_return) + self.mean_return
            else:
                clis = is_weights * self.discounted_return(rewards)

        ESS = ((is_weights.sum()) ** 2 / (((is_weights) ** 2).sum())).item()
        return clis, ESS, is_weights

    def test_nll(self, replay_buffer, is_validation=False):
        states, actions, rewards, not_dones, pibs, _, nn_action_dist = replay_buffer.sample(replay_buffer.max_size)
        with torch.no_grad():
            log_prob = not_dones * ((self.policy(states) * pibs).sum(dim=-1,keepdim=True))
        if is_validation:
            print(f"Validation Log Likelihood: {log_prob.mean().item():.3f}")
        else:
            print(f"Train Log Likelihood: {log_prob.mean().item():.3f}")

    def set_cv(self, replay_buffer):
        states, actions, rewards, not_dones, pibs, estm_pibs, nn_action_dist = replay_buffer.sample(
            replay_buffer.max_size)
        if self.cv_type == "const":
            self.mean_return = 50
        elif self.cv_type == "const_mean":
            self.mean_return = self.discounted_return(rewards).mean().item()
        return

    def test_var(self, replay_buffer, is_validation=False):
        states, actions, rewards, not_dones, pibs, estm_pibs, nn_action_dist = replay_buffer.sample(replay_buffer.max_size)
        with torch.no_grad():
            clis, ESS, weights = self.compute_dr(states, actions, rewards, not_dones, pibs, nn_action_dist, estm_pibs, eval_mode=True)
            mean_is = clis.mean().item()
            var_is = clis.var().item()

        A = - mean_is / np.sqrt(var_is)
        B = 0.5 / np.sqrt(var_is)
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B

        with torch.no_grad():
            wclis, wESS, weights = self.compute_dr(states, actions, rewards, not_dones, pibs, nn_action_dist,
                                                   estm_pibs, weighted=True, eval_mode=True)
            mean_wis = wclis.mean().item()
            var_wis = wclis.var().item()

        if is_validation:
            print(f"Valid WIS: {mean_wis:.3e}, ESS: {wESS:4.1f}, Var(WIS): {var_wis:.3e}, IS: {mean_is:.3e},  ESS: {ESS:4.1f}, Var(IS): {var_is:.3e}")
            return mean_wis, var_wis, wESS
        else:
            print(f"Train WIS: {mean_wis:.3e}, ESS: {wESS:4.1f}, Var(WIS): {var_wis:.3e}, IS: {mean_is:.3e},  ESS: {ESS:4.1f}, Var(IS): {var_is:.3e}")
            return mean_wis, var_wis, wESS

    def behavior_cloning(self, replay_buffer, batch_size):
        states, actions, rewards, not_dones, pibs, _, nn_action_dist = replay_buffer.sample(batch_size)

        bc_loss = self.bc_loss(states, actions, not_dones, pibs)
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
        self.iteration += 1

    def train(self, replay_buffer, batch_size):
        #torch.autograd.set_detect_anomaly(True)

        states, actions, rewards, not_dones, pibs, _, nn_action_dist = replay_buffer.sample(batch_size)

        pg_loss = self.pg_loss(states, actions, rewards, not_dones, pibs, nn_action_dist)
        policy_loss = pg_loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        for name, params in self.policy.named_parameters():
            if params.grad is not None and torch.isnan(params.grad).any():
                print(f"ValueError: nan in Policy {name}'s gradient. Ending current train loop and updating target.")
                self.iteration += 1
                return
        self.policy_optimizer.step()
        self.iteration += 1

    def get_prob(self, state, pibs, nn_action_dist):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape((-1, self.state_dim)).to(self.device)
            prob = self.policy(state).exp()
            pibs = torch.FloatTensor(pibs).reshape(-1, self.num_actions).to(self.device)
            nn_action_dist = torch.FloatTensor(nn_action_dist).reshape(-1, self.num_actions).to(self.device)
            if self.action_mask_type == "step" or self.action_mask_type == "trajectory":
                mask = (pibs >= self.threshold).float()
            elif self.action_mask_type == "nn_action_dist":
                mask = (nn_action_dist <= self.threshold).float()
            if mask.sum() > 0:
                prob = prob * mask
            prob = prob / prob.sum().clamp(1e-20,1)
        return prob[0].cpu().numpy()

    def select_action(self, state, pibs, nn_action_dist):
        prob = self.get_prob(state, pibs, nn_action_dist)
        a = np.random.choice(self.num_actions, p=prob)
        return a

    def reset_optimizer(self):
        self.policy_optimizer = getattr(torch.optim, self.optimizer)(self.policy.parameters(), **self.optimizer_parameters)

    def save(self, filename):
        self.policy.save(filename+"policy.pth")

    def load(self, filename):
        self.policy.load(filename + "policy.pth")
