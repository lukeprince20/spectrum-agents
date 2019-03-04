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


class ReplicatedQ(Agent):
    def __init__(self, agent_id, env, seed=None, start=None):
        N = len(env.channels)
        super().__init__(agent_id, N, seed=seed, start=start)
        self.env = env
        #self.ns = np.prod(space.n for space in env.state_space.spaces) # 2 ** N
        #self.na = N
        #self.q = np.zeros((self.ns, self.na)) # this is sort of a shortcut but will work
        #self.belief_dict = {i:np.random.random_sample(m.shape)
        #        for i,m in env.transition_dict.items()}
        self.ns = tuple(space.n for space in env.state_space.spaces)
        self.na = tuple(space.n for space in env.action_space.spaces)
        self.q_dict = {i:np.zeros((ns,na))
                for i, (ns,na) in enumerate(zip(self.ns, self.na))}
        #self.q_dict = {i:np.random.random_sample((ns,na))
        #        for i, (ns,na) in enumerate(zip(self.ns, self.na))}
        self.belief_dict = {i:np.random.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v))
                for i,v in self.belief_dict.items())

        # Q-learning parameters
        self.eps = 1.0      # epsilon
        self.eps_schedule = lambda eps: max(eps-0.025, 0.05)
        self.lr = 0.2       # alpha
        self.discount = 0.2 # gamma

    def _learn(self,o=None,r=None):
        #def tuple_to_state(s_tuple):
        #    return sum(2**n * v for n,v in enumerate(s_tuple))

        def belief_update(b,o,p):
            #return np.dot(p.transpose(), b) if o is None else p[o,:]
            if o is None:
                belief = np.dot(p.transpose(), b) # Markov evolution
            elif o is 0:
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o is 1:
                belief = np.array([0.0, 1.0]) # observed as occupied
            else:
                raise ValueError("unexpected observation value '{}'".format(o))
            return belief

        # determine belief vector for next state
        saor_dict = {k:(s,a,o,r) for k,(s,a,o,r) in enumerate(zip(self.s, self.a, o, r))}
        next_belief_dict = {k:
                belief_update(b, saor_dict[k][2], self.env.transition_dict[k])
                for (k,b) in self.belief_dict.items()}

        # calculate Q-table delta
        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in next_belief_dict.items()}
        greedy_policy_dict = {k:np.max(a) for (k,a) in expected_q_dict.items()}

        delta_q_dict = {k:
                self.lr * (reward
                    + self.discount * greedy_policy_dict[k]
                    - self.q_dict[k][s,a])
                for k,(s,a,_,reward) in saor_dict.items()}

        # update Q-table
        for k, (s,a,_,_) in saor_dict.items():
            self.q_dict[k][s,a] = self.q_dict[k][s,a] + delta_q_dict[k]

        # update belief vector
        self.belief_dict.update(next_belief_dict)

    def _decide(self,o=None,r=None):
        decision_dict = self.__egreedy(o,r)
        return tuple(decision_dict[k] for k in range(self.N))

    def __egreedy(self,o=None,r=None):
        def egreedy_element(o, q, isGreedy):
            return np.argmax(q[o,:]) if isGreedy else random.randint(0,1)

        self.eps = self.eps_schedule(self.eps)
        o_dict = {k:v for k,v in enumerate(o)}
        greedy_dict = {k:random.random() > self.eps for k in self.q_dict.keys()}
        return {k:egreedy_element(o[k], self.q_dict[k], tf) for k, tf in greedy_dict.items()}
