import numpy as np
import random

from spectrum_agents import Agent

class Q(Agent):
    """
    Q-learning Agent inheriting from Agent base class.
    Updates Q-table and e-greedily picks new actions.
    """
    def __init__(self, agent_id, N, seed=None, start=None):
        super().__init__(agent_id, N, seed=seed, start=start)
        self.ns = 2 ** N # number of states
        self.na = N # number of decisions
        self.q = np.zeros((self.ns, self.na))

        # Q-learning parameters
        self.eps = 1.0 # epsilon
        self.eps_min = 0.01
        self.lr = 0.7 # alpha
        self.discount = 0.4 # gamma
        self.eps_schedule = lambda eps: max(eps-1e-2, self.eps_min)

        self.temp = 1.0 # temperature

    def _decide(self,o=None,r=None):
        return self._egreedy(o,r)
#        return self._softmax(o,r)

    def _learn(self,o=None,r=None):
        # Q-table update
        # q(s,a) := q(s,a) + alpha * [r(s,a) + gamma * max[q(s,:)] - q(s,a)]
        if self.s is not None:
            delta = r + \
                    self.discount * np.max(self.q[self.s,:]) - \
                    self.q[self.s,self.a]
            self.q[self.s, self.a] += self.lr * delta

    def _egreedy(self,o=None,r=None):
        self.eps = self.eps_schedule(self.eps)
        if random.random() < self.eps:
            return random.randint(0,self.na-1)
        else:
            return np.argmax(self.q[o,:])

    def _softmax(self,o=None,r=None):
        pdf = np.exp(self.q[o,:]/self.temp)
        pdf = pdf/sum(pdf)
        return np.random.choice(self.na, p=pdf)
