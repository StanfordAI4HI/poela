import numpy as np
import matplotlib.pyplot as plt
""" 
Code modified based on Havard DTAK group's implmentation: 
https://github.com/dtak/interpretable_ope_public/blob/master/rl_basics_local/domains/cancer/cancer_env.py
"""


class CancerSim:
    # Meta data of the simulator
    meta_parameters = {
        "PCV": {
            "P0_m": 7.13,
            "P0_se": 0.25,
            "Q0_m": 41.2,
            "Q0_se": 0.07,
            "lambda_p_m": 0.121,
            "lambda_p_se": 0.16,
            "K_pq_m": 0.0295,
            "K_pq_se": 0.21,
            "K_qpp_m": 0.0031,
            "K_qpp_se": 0.35,
            "delta_qp_m": 0.00867,
            "delta_qp_se": 0.21,
            "gamma_m": 0.729,
            "gamma_se": 0.37,
            "kde_m": 0.24,
            "kde_se": 0.33,
        },
        "TMZ": {
            "P0_m": 0.924,
            "P0_se": 0.57,
            "Q0_m": 42.3,
            "Q0_se": 0.08,
            "lambda_p_m": 0.114,
            "lambda_p_se": 0.29,
            "K_pq_m": 0.0226,
            "K_pq_se": 0.54,
            "K_qpp_m": 0.0045,
            "K_qpp_se": 0.70,
            "delta_qp_m": 0.0214,
            "delta_qp_se": 0.34,
            "gamma_m": 0.842,
            "gamma_se": 0.43,
            "kde_m": 0.32,
            "kde_se": 0.34,
        },
        "RT": {
            "P0_m": 3.89,
            "P0_se": 0.28,
            "Q0_m": 40.3,
            "Q0_se": 0.06,
            "lambda_p_m": 0.138,
            "lambda_p_se": 0.16,
            "K_pq_m": 0.0249,
            "K_pq_se": 0.41,
            "K_qpp_m": 0.0,
            "K_qpp_se": 0.0,
            "delta_qp_m": 0.0125,
            "delta_qp_se": 0.29,
            "gamma_m": 1.71,
            "gamma_se": 0.24,
            "kde_m": 0.317,
            "kde_se": 0.60,
        },
    }

    def __init__(self, dose_penalty=0.0, therapy_type="PCV", a_bins=1, env_seed=None, max_steps=30, state_dim=3,
                 reward_type="sparse"):
        ''' Intitilize patient parameters '''

        self.rng = np.random.default_rng(env_seed)

        self.P0, self.Q0, self.lambda_p, self.k_pq, self.k_qpp, self.delta_qp, self.gamma, self.kde \
            = self._patient_parameter_generator(therapy_type, self.rng)

        self.type = therapy_type
        self.k = 100
        self.dose_penalty = dose_penalty
        self.max_steps = max_steps
        self.state = None
        self.time_step = None
        self.state_dim = state_dim
        self.num_actions = a_bins + 1
        self.reward_type = reward_type

        self.reset()

    @staticmethod
    def _patient_parameter_generator(therapy_type, rng):
        p = CancerSim.meta_parameters[therapy_type]
        # P0 = p["P0_m"] + rng.normal(scale=p["P0_se"]*p["P0_m"])
        # Q0 = p["Q0_m"] + rng.normal(scale=p["Q0_se"]*p["Q0_m"])
        # lambda_p = p["lambda_p_m"] + rng.normal(scale=p["lambda_p_se"]*p["lambda_p_m"])
        # K_pq = p["K_pq_m"] + rng.normal(scale=p["K_pq_se"]*p["K_pq_m"])
        # K_qpp = p["K_qpp_m"] + rng.normal(scale=p["K_qpp_se"]*p["K_qpp_m"])
        # delta_qp = p["delta_qp_m"] + rng.normal(scale=p["delta_qp_se"]*p["delta_qp_m"])
        # gamma = p["gamma_m"] + rng.normal(scale=p["gamma_se"]*p["gamma_m"])
        # kde = p["kde_m"] + rng.normal(scale=p["kde_se"]*p["kde_m"])

        P0 = p["P0_m"] * rng.lognormal(sigma=0.03)
        Q0 = p["Q0_m"] * rng.lognormal(sigma=0.03)
        lambda_p = p["lambda_p_m"] * rng.lognormal(sigma=0.03)
        K_pq = p["K_pq_m"] * rng.lognormal(sigma=0.03)
        K_qpp = p["K_qpp_m"] * rng.lognormal(sigma=0.03)
        delta_qp = p["delta_qp_m"] * rng.lognormal(sigma=0.03)
        gamma = p["gamma_m"] * rng.lognormal(sigma=0.03)
        kde = p["kde_m"] * rng.lognormal(sigma=0.03)
        return P0, Q0, lambda_p, K_pq, K_qpp, delta_qp, gamma, kde

    def reset(self):
        C = 0
        Q_p = 0

        self.P0, self.Q0, self.lambda_p, self.k_pq, self.k_qpp, self.delta_qp, self.gamma, self.kde \
            = self._patient_parameter_generator(self.type, self.rng)

        P = self.P0
        Q = self.Q0

        self.state = np.array([C, P, Q, Q_p])
        self.time_step = 0
        return self.observe(self.state)

    def is_done(self):
        return self.time_step >= self.max_steps

    def observe(self, state):
        if self.state_dim == 4:
            return state
        elif self.state_dim == 5:
            return np.concatenate((state, np.array([self.time_step])))
        elif self.state_dim == 3:
            return np.array([state[0], state[1:4].sum(), self.time_step])

    def step(self, action):
        C, P, Q, Q_p = self.state
        P_star = P + Q + Q_p

        if self.type == "TMZ":
            C = float(action/(self.num_actions-1))
        else:
            C += float(action / (self.num_actions-1))
            C = C - self.kde * C

        P = (P + self.lambda_p * P * (1-P_star/self.k) + self.k_qpp * Q_p
             - self.k_pq * P - self.gamma * C * self.kde * P)
        Q = Q + self.k_pq * P - self.gamma * C * self.kde * Q
        Q_p = (Q_p + self.gamma * C * self.kde * Q - self.k_qpp * Q_p
               - self.delta_qp * Q_p)

        next_state = np.array([C, P, Q, Q_p])
        self.state = next_state
        P_star_new = P + Q + Q_p

        self.time_step += 1
        # reward = P_star - P_star_new - self.dose_penalty * C
        if self.reward_type == "dense":
            reward = P_star - P_star_new - self.dose_penalty * C
        else:
            if self.is_done():
                reward = (self.P0 + self.Q0 - P_star_new) - self.dose_penalty * C
            else:
                reward = - self.dose_penalty * C

        return self.observe(next_state), reward, self.is_done(), {}


class PolicyCancer:
    def __init__(self, months_for_treatment=9, eps_behavior=0.3):
        self.num_actions = 2
        self.months_for_treatment = months_for_treatment
        if months_for_treatment is None:
            self.months_for_treatment = np.random.random_integers(5, 13)
        self.eps_behavior = eps_behavior
        if eps_behavior is None:
            self.eps_behavior = np.random.beta(3, 11)

    def __call__(self, state, time_step):
        if np.random.rand() < self.eps_behavior and time_step > 0:
            return np.array([np.random.choice(2)])
        if time_step < self.months_for_treatment:
            return np.array([1])
        else:
            return np.array([0])

    def return_probs(self, state, time_step):
        if time_step >= self.months_for_treatment:
            return np.array([1.0 - self.eps_behavior/2, self.eps_behavior/2])
        else:
            return np.array([self.eps_behavior / 2, 1.0 - self.eps_behavior / 2])
