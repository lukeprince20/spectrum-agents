import numpy as np
import random

from spectrum_agents import Agent

class ReplicatedQ(Agent):
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        super().__init__(agent_id, env, sensors=sensors, seed=seed, start=start)
        self.ns = tuple(space.n for space in env.state_space.spaces)
        self.na = tuple(space.n for space in env.action_space.spaces)
        self.q_dict = {i:np.zeros((ns,na))
                for i, (ns,na) in enumerate(zip(self.ns, self.na))}
        self.belief_dict = {i:np.random.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v))
                for i,v in self.belief_dict.items())

        # Q-learning parameters
        self.eps = 1.0      # epsilon
        self.eps_schedule = lambda eps: max(eps-0.025, 0.05)
        self.lr = 0.2       # alpha
        self.discount = 0.2 # gamma

    def learn(self, observation, reward):
        #def tuple_to_state(s_tuple):
        #    return sum(2**n * v for n,v in enumerate(s_tuple))

        def belief_update(b,o,p):
            if o is None:
                belief = np.dot(p.transpose(), b) # Markov evolution
            elif o is 0:
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o is 1:
                belief = np.array([0.0, 1.0]) # observed as occupied
            return belief

        # determine belief vector for next state
        saor_dict = {k:(s,a,o,r) for k,(s,a,o,r)
                in enumerate(zip(self.state, self.action, observation, reward))}
        next_belief_dict = {k:
                belief_update(b, saor_dict[k][2], self.env.transition_dict[k])
                for (k,b) in self.belief_dict.items()}

        # calculate Q-table delta
        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in next_belief_dict.items()}
        greedy_policy_dict = {k:np.max(a) for (k,a) in expected_q_dict.items()}

        delta_q_dict = {k:
                self.lr * (r
                    + self.discount * greedy_policy_dict[k]
                    - self.q_dict[k][s,a])
                for k,(s,a,_,r) in saor_dict.items()}

        # update Q-table
        for k, (s,a,_,_) in saor_dict.items():
            self.q_dict[k][s,a] = self.q_dict[k][s,a] + delta_q_dict[k]

        # update belief vector
        self.belief_dict.update(next_belief_dict)

    def decide(self, observation, reward):
        decision_dict = self._egreedy(observation)
        return tuple(decision_dict[k] for k in range(len(self.env.channels)))

    def _egreedy(self, observation):
        def egreedy_element(o, q, isGreedy):
            return np.argmax(q[o,:]) if isGreedy else random.randint(0,1)

        self.eps = self.eps_schedule(self.eps)
        o_dict = {k:v for k,v in enumerate(observation)}
        greedy_dict = {k:random.random() > self.eps for k in self.q_dict.keys()}
        decision_dict = {k:
                egreedy_element(observation[k], self.q_dict[k], tf)
                for k, tf in greedy_dict.items()}
        return decision_dict
