#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
import numpy as np

from spectrum_agents import Agent

class Genie(Agent):
    def __init__(self, agent_id, env, seed=None, start=None):
        N = len(env.channels)
        super().__init__(agent_id, N, seed=seed, start=start)
        self.env = env
        self.decision_dict = {}
        for i,m in env.transition_dict.items():
            v = np.zeros(m.shape)
            for j in range(m.shape[0]):
                k = np.argmax(m[j,:])
                v[j,k] = 1
            self.decision_dict[i] = v

    def _decide(self,o=None,r=None):
        return tuple(np.argmax(x[s,:]) for s,x in zip(o, self.decision_dict.values()))

    def _learn(self,o=None,r=None):
        pass


class BeliefGenie(Agent):
    def __init__(self, agent_id, env, seed=None, start=None):
        N = len(env.channels)
        super().__init__(agent_id, N, seed=seed, start=start)
        self.env = env
        #self.belief_dict = {i:np.random.random_sample(m.shape) 
        #        for i,m in env.transition_dict.items()}
        self.belief_dict = {i:np.random.random_sample(shape.n)
                for i,shape in enumerate(env.state_space.spaces)}
        self.belief_dict.update((i,v/sum(v)) 
                for i,v in self.belief_dict.items())

    def _learn(self,o=None,r=None):
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

        observation_dict = {k:v for k,v in enumerate(o)}
        next_belief_dict = {k:
                belief_update(b, observation_dict[k], self.env.transition_dict[k]) 
                for (k,b) in self.belief_dict.items()}
        self.belief_dict.update(next_belief_dict);

    def _decide(self,o=None,r=None):
        decision_dict = {k:np.argmax(v) for k,v in self.belief_dict.items()}
        return tuple(decision_dict[k] for k in range(self.N))
