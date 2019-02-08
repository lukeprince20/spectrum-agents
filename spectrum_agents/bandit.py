from spectrum_agents import Agent
import numpy as np
from math import log, sqrt

class Bandit(Agent):
    """
    Multi-armed bandit agent.
    Chooses actions minimizing regret.
    using ucb1.
    """
    def __init__(self, agent_id, N, **kwargs):
        super().__init__(agent_id, N, **kwargs)
        self.avg_arm_reward = np.zeros(N)
        self.n_arms = np.zeros(N)
        self.n_epoch = 0

        # Handle special case of initial starting decision
        if self._start is not None:
            self.n_epoch +=1
            self.n_arms[self._start] += 1

    def _decide(self,o=None,r=None):
        decision = self._ucb1()
        self.n_epoch += 1
        self.n_arms[decision] += 1
        return decision

    def _learn(self,o=None,r=None):
        # update cumulative reward average for arm
        if r is not None:
            self.avg_arm_reward[self.a] += \
                    (r - self.avg_arm_reward[self.a]) / self.n_arms[self.a]

    def _ucb1(self):
        """
        Version of the Upper Confidence Bound (UCB) algorithm for 
        MAB arm selection.
        """
        if self.n_epoch < self.N:
            return self.n_epoch # choose unexplored arm choice
        else:
            arm_confidences = [x + sqrt(2*log(self.n_epoch)/y)
                            for x,y in zip(self.avg_arm_reward, self.n_arms)]
            return np.argmax(arm_confidences)
