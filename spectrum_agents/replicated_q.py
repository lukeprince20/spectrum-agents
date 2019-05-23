import numpy as np
from spectrum_agents import Agent
import pdb

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
            elif o == 0:
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o == 1:
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

        next_belief_dict = {k:
                np.dot(self.env.transition_dict[k].transpose(), b)
                for (k,b) in self.belief_dict.items()}

        idle_belief = tuple(b[0] for _,b in next_belief_dict.items())
        idle_likelihood = tuple(x >= 0.5 for x in idle_belief)
        idle_belief_arg_sorted = argsort_randomtie(idle_belief)
        idle_belief_arg_sorted_desc = idle_belief_arg_sorted[::-1]

        # exploration - epsilon-greedy action selection
        self.eps = self.eps_schedule(self.eps)
        if (self.rng.random_sample() < self.eps):
            random_decision = np.zeros(len(self.env.channels), np.int64)
            sensor_index = self.rng.choice(len(self.env.channels), self.sensors, replace=False)
            #random_decision[sensor_index] = 1
            random_decision[sensor_index] = self.rng.choice([0, 1], self.sensors)
            return random_decision
            #return tuple(a.item() for a in random_decision)

        # exploitation - q-learning action selection
        expected_q_dict = {k:b.dot(self.q_dict[k]) for (k,b) in self.belief_dict.items()}
        idle_q = tuple(v[0] for _,v in expected_q_dict.items())
        idle_q_args_sorted = argsort_randomtie(idle_q)
        idle_q_args_sorted_desc = idle_q_args_sorted[::-1]
        
        # limit used sensors to channels with idle likelihood
        num_active_sensors = min(sum(idle_likelihood), self.sensors)
        active_sensors = idle_q_args_sorted_desc[:num_active_sensors]

        # populate sensor decision array
        sensor_decision = np.zeros(len(self.env.channels), np.int64)
        sensor_decision[active_sensors] = 1
        return tuple(a.item() for a in sensor_decision)
