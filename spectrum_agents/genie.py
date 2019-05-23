#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
import numpy as np
from spectrum_agents import Agent
import pdb

class Genie(Agent):
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        super().__init__(agent_id, env, sensors=sensors, seed=seed, start=start)

        self.decision_dict = {}
        for i,m in env.transition_dict.items():
            v = np.zeros(m.shape)
            for j in range(m.shape[0]):
                k = np.argmax(m[j,:])
                v[j,k] = 1
            self.decision_dict[i] = v

    def decide(self, observation, reward):
        observation = tuple(0 if (o is None) else o for o in observation)
        decision = tuple(1 if (np.argmax(p[o,:])==0) else 0
                for o,p in zip(observation, self.decision_dict.values()))

        # make decision that maximises reward for num of sensors
        best_indices = np.argsort(decision)
        best_indices_desc = best_indices[::-1]
        active_sensors = best_indices_desc[:self.sensors]
        sensor_decision = np.zeros(len(self.env.channels), np.int64)
        full_decision = np.array(decision, np.int64)
        sensor_decision[active_sensors] = full_decision[active_sensors]

        return sensor_decision

#    def decide(self, observation, reward):
#        observation = tuple(0 if (o is None) else o for o in observation)
#        decision = tuple(1 if (np.argmax(p[o,:])==0) else 0
#                for o,p in zip(observation, self.decision_dict.values()))
#        return decision

    def learn(self, observation, reward):
        pass


class BeliefGenie(Agent):
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        super().__init__(agent_id, env, sensors=sensors, seed=seed, start=start)

        self.belief_dict = {i:self.rng.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v)) 
                for i,v in self.belief_dict.items())

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
        next_belief_dict = {k:
                belief_update(b, observation[k], self.env.transition_dict[k]) 
                for (k,b) in self.belief_dict.items()}
        self.belief_dict.update(next_belief_dict);

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

        # limit used sensors to channels with idle likelihood
        num_active_sensors = min(sum(idle_likelihood), self.sensors)
        active_sensors = idle_belief_arg_sorted_desc[:num_active_sensors]

        # populate sensor decision array
        sensor_decision = np.zeros(len(self.env.channels), np.int64)
        sensor_decision[active_sensors] = 1

        return tuple(a for a in sensor_decision)
