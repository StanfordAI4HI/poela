import numpy as np


class ISFail(object):
    def __init__(self, K=10):
        self.s = None
        self.K = K

    def start(self):
        return (0, 0)

    def transition(self, s, a):
        if s[0] == 0:
            if a == 0:
                return (1, 0), 0, False
            elif a == 2:
                return (1, 1), 0, False
            elif a == 1:
                return (1, 2+np.random.binomial(1, 0.5)), 0, False

        if s == (1, 0):
            return (1, 0), 2*np.random.binomial(1, 0.5)-1, True

        if s == (1, 1):
            return (1, 1), -5.0, True

        if s == (1, 2):
            return (1, 2), 1, True

        if s == (1, 3):
            return (2, a), -5.0, True

    def reset(self):
        self.s = self.start()
        return self.s

    def make_obs(self, s):
        if s == (1, 1):
            return (1, 0)
        else:
            return s

    def step(self, a):
        ns, r, done = self.transition(self.s, a)
        self.s = ns
        return self.make_obs(ns), r, done

    def get_actions(self, obs):
        if obs == (0, 0):
            return [0, 1]
        if obs == (1, 0):
            return [0]
        # if obs == (1, 1):
        #     return [0]
        if obs == (1, 2):
            return [0]
        else:
            return [a for a in range(self.K)]

    def get_obs(self):
        obs = [
            (0, 0),
            (1, 0),
            (1, 2),
            (1, 3),
        ]
        return obs

    def get_episode(self, policy):
        episodes = []
        obs = self.reset()
        done = False
        while not done:
            a_list = self.get_actions(obs)
            pa_list = policy[obs]
            a = np.random.choice(a_list, p=pa_list)
            pa = pa_list[int(a)]
            nobs, r, done = self.step(a)
            episodes.append((obs,a,r,done,pa))
            obs = nobs
        return episodes

    def get_uniform_policy(self):
        policy = dict()
        obss = self.get_obs()
        for obs in obss:
            na = len(self.get_actions(obs))
            policy[obs] = [1/na]*na
        return policy

    def get_all_policies(self):
        all_policies = []
        for a00 in range(2):
            for a21 in range(self.K):
                # n = int(pow(self.K, self.K))
                # for i in range(n):
                policy = {
                    (0, 0): [float(a == a00) for a in range(3)],
                    (1, 0): [1.0],
                    (1, 2): [1.0],
                    (1, 3): [float(a == a21) for a in range(self.K)]
                }
                # for s2 in range(self.K):
                #     a22 = int(i / pow(self.K, s2)) % self.K
                #     policy[(2,s2)] = [float(a == a22) for a in range(self.K)]
                all_policies.append(policy)
        return all_policies


def evaluate(policy, dataset):
    w_list = []
    r_list = []
    for episode in dataset:
        w = 1
        for sample in episode:
            obs,a,r,done,pib = sample
            pie = policy[obs][a]
            w *= pie/pib
            if done:
                w_list.append(w)
                r_list.append(r)
    return np.array(w_list), np.array(r_list)


def get_objective(weights, rewards, weighted, var_coeff):
    if weighted:
        weights /= weights.mean()
        estm = (weights*rewards).mean()
        var_pen = np.sqrt(np.sum((weights**2)*((rewards-estm)**2))/(np.sum(weights))**2)
    else:
        var_pen = (weights*rewards).std()/np.sqrt(len(weights))
    objective = (weights*rewards).mean() - var_coeff*var_pen
    # print(objective, var_pen)
    return max(objective,-5), var_pen


def get_counts(env, dataset):
    counts = dict()
    s_counts = dict()
    for obs in env.get_obs():
        s_counts[obs] = 0
        for action in env.get_actions(obs):
            counts[(obs, action)] = 0

    for episode in dataset:
        for sample in episode:
            obs, a, _, _, _ = sample
            counts[(obs, a)] += 1
            s_counts[obs] += 1

    return counts, s_counts


def check_policy(counts, x_counts, policy, threshold=1.0):
    for (obs,a) in counts.keys():
        if x_counts[obs] > 0 and counts[(obs, a)] < threshold and policy[obs][a] > 0:
            return False
    return True


def main():
    A = 2
    H = 3
    K = A**H
    for n in [8, 16, 32, 64, 128, 256, 512]: #1024, 2048, 4096
        print("n=", n)
        avg0 = []
        avg1 = []
        avg2 = []
        for n_e in range(100):
            var_coeffs = [0, 0.1, 1, 10, 100]
            env = ISFail(K)
            pib = env.get_uniform_policy()
            dataset = [env.get_episode(pib) for i in range(n)]
            Pi = env.get_all_policies()
            xa_counts, x_counts = get_counts(env, dataset)
            best_result = dict()
            best_policy = dict()
            for pie in Pi:
                w, r = evaluate(pie, dataset)
                for weighted in [True, False]:
                    for var_coeff in var_coeffs:
                        result, var_pen = get_objective(w, r, weighted, var_coeff)
                        key = (0, weighted, var_coeff)
                        if (key not in best_result.keys()) or (result >= best_result[key]):
                            best_result[key] = result
                            best_policy[key] = pie
                        if check_policy(xa_counts, x_counts, pie):
                            key = (1, weighted, var_coeff)
                            if (key not in best_result.keys()) or (result >= best_result[key]):
                                best_result[key] = result
                                best_policy[key] = pie
            flag0 = False
            flag1 = False
            flag2 = False
            for key in best_policy.keys():
                # print(key, best_policy[key][(0, 0)][0]) #best_policy[key][(0, 0)][0]*1+best_policy[key][(0, 0)][2]*(1.1-1)/2, best_result[key]
                if key[0] == 0 and key[1] == False:
                    flag0 = flag0 or best_policy[key][(0, 0)][0]
                if key[0] == 0 and key[1] == True:
                    flag1 = flag1 or best_policy[key][(0, 0)][0]
                if key == (1, True, 0):
                    flag2 = flag2 or best_policy[key][(0, 0)][0]
            avg0.append(flag0)
            avg1.append(flag1)
            avg2.append(flag2)
        print(np.mean(avg0), np.mean(avg1), np.mean(avg2))


if __name__ == "__main__":
    main()





