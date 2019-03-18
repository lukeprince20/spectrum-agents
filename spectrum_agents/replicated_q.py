import numpy as np
from spectrum_agents import Agent

class ReplicatedQ(Agent):
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        super().__init__(agent_id, env, sensors=sensors, seed=seed, start=start)
        self.ns = tuple(space.n for space in env.state_space.spaces)
        self.na = tuple(space.n for space in env.action_space.spaces)
        self.q_dict = {i:np.zeros((ns,na))
                for i, (ns,na) in enumerate(zip(self.ns, self.na))}
        self.belief_dict = {i:self.rng.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v))
                for i,v in self.belief_dict.items())

        # Q-learning parameters
        self.eps = 1.0      # epsilon
        self.eps_schedule = lambda eps: max(eps-0.025, 0.05)
        self.lr = 0.2       # alpha
        self.discount = 0.2 # gamma

    def learn(self, observation, reward):
        def belief_update(b,o,p):
            if o is None:
                belief = np.dot(p.transpose(), b) # Markov evolution
            elif o is 0:
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o is 1:
                belief = np.array([0.0, 1.0]) # observed as occupied
            return belief

        observation = self.partial_observation(observation)

        # determine belief vector for next state
        next_belief_dict = {k:
                belief_update(b, observation[k], self.env.transition_dict[k])
                for (k,b) in self.belief_dict.items()}

        # calculate expected-Q and greedy action selection
        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in next_belief_dict.items()}
        greedy_policy_dict = {k:np.max(a) for (k,a) in expected_q_dict.items()}
        for k,a in enumerate(self.action):
            # calculate Q-table delta
            delta_q = self.lr * self.belief_dict[k] * (reward[k]
                    + self.discount * greedy_policy_dict[k]
                    - self.q_dict[k][:,a])
            # update Q-table
            self.q_dict[k][:,a] = self.q_dict[k][:,a] + delta_q

        # update belief vector
        self.belief_dict.update(next_belief_dict)

    def decide(self, observation, reward):
        def argsort_randomtie(x):
            r = self.rng.random_sample(len(x))
            return np.lexsort((r,x))

        observation = self.partial_observation(observation)

        self.eps = self.eps_schedule(self.eps)
        if (self.rng.random_sample() < self.eps):
            random_decision = np.zeros(len(self.env.channels), np.int64)
            sensor_index = self.rng.choice(len(self.env.channels), self.sensors, replace=False)
            random_decision[sensor_index] = 1
            return random_decision

        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in self.belief_dict.items()}
        idle_q = tuple(v[0] for _,v in expected_q_dict.items())
        q_sorted = argsort_randomtie(idle_q)
        sensor_decision = np.zeros(len(self.env.channels), np.int64)
        sensor_decision[q_sorted[::-1][:self.sensors]] = 1
        return tuple(a for a in sensor_decision)

        #decision_dict = self._egreedy(observation)
        #decision = tuple(decision_dict[k] for k in range(len(self.env.channels)))
        #return decision
        ##return tuple(a if s else None for s,a in zip(sensor_decision, decision))

    def _egreedy(self, observation):
        def egreedy_element(qb, isGreedy):
            return np.argmax(qb) if isGreedy else self.rng.randint(0,2)

        self.eps = self.eps_schedule(self.eps)
        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in self.belief_dict.items()}
        greedy_dict = {k:self.rng.random_sample() > self.eps for k in self.q_dict.keys()}
        decision_dict = {k:
                egreedy_element(expected_q_dict[k], tf)
                for k, tf in greedy_dict.items()}
        return decision_dict
