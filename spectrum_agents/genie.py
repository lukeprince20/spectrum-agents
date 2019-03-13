#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
import numpy as np

from spectrum_agents import Agent

class Genie(Agent):
    def __init__(self, agent_id, env, seed=None, start=None):
        super().__init__(agent_id, env, sensors=len(env.channels), seed=seed, start=start)

        self.decision_dict = {}
        for i,m in env.transition_dict.items():
            v = np.zeros(m.shape)
            for j in range(m.shape[0]):
                k = np.argmax(m[j,:])
                v[j,k] = 1
            self.decision_dict[i] = v

    def decide(self, observation=None, reward=None):
        decision = tuple(np.argmax(p[o,:]) 
                for o,p in zip(observation, self.decision_dict.values()))
        return decision

    def learn(self, observation=None, r=None):
        pass


class BeliefGenie(Agent):
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        super().__init__(agent_id, env, sensors=sensors, seed=seed, start=start)

        self.belief_dict = {i:np.random.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v)) 
                for i,v in self.belief_dict.items())

    def learn(self, observation, reward):
        def belief_update(b,o,p):
            if o is None:
                belief = np.dot(p.transpose(), b) # Markov evolution
            elif o is 0: 
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o is 1:
                belief = np.array([0.0, 1.0]) # observed as occupied 
            return belief

        next_belief_dict = {k:
                belief_update(b, observation[k], self.env.transition_dict[k]) 
                for (k,b) in self.belief_dict.items()}
        self.belief_dict.update(next_belief_dict);

    def decide(self,observation, reward):
        decision_dict = {k:np.argmax(v) for k,v in self.belief_dict.items()}
        decision = tuple(decision_dict[k] for k in range(len(self.env.channels)))
        return decision
