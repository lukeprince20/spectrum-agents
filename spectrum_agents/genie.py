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

    def decide(self, observation, reward):
        observation = tuple(0 if (o is None) else o for o in observation)
        decision = tuple(1 if (np.argmax(p[o,:])==0) else 0
                for o,p in zip(observation, self.decision_dict.values()))
        return decision

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
            elif o is 0: 
                belief = np.array([1.0, 0.0]) # observed as vacant
            elif o is 1:
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

        ##max_belief = tuple(np.max(v) for _,v in self.belief_dict.items())
        ##b_sorted = argsort_randomtie(max_belief)
        ##sensor_decision = np.zeros(len(self.env.channels), bool)
        ##sensor_decision[b_sorted[::-1][:self.sensors]] = True

        observation = self.partial_observation(observation)

        next_belief_dict = {k:
                np.dot(self.env.transition_dict[k].transpose(), b)
                for (k,b) in self.belief_dict.items()}
        #print("next_belief_dict {}".format(next_belief_dict))
        idle_belief = tuple(b[0] for _,b in next_belief_dict.items())
        #print("idle_belief {}".format(idle_belief))
        b_sorted = np.argsort(idle_belief)
        #print("idle_belief_sorted {}".format(b_sorted))
        b_sorted = argsort_randomtie(idle_belief)
        #print("idle_belief_sorted {}".format(b_sorted))
        sensor_decision = np.zeros(len(self.env.channels), np.int64)
        #print("idle_belief_sorted[::-1]: {}".format(b_sorted[::-1]))
        sensor_decision[b_sorted[::-1][:self.sensors]] = 1
        #print("sensor_decision {}".format(sensor_decision))
        return tuple(a for a in sensor_decision)

        ##decision_dict = {k:np.argmax(v) for k,v in self.belief_dict.items()}
        #decision_dict = {k:np.argmin(v) for k,v in self.belief_dict.items()}
        #decision = tuple(decision_dict[k] for k in range(len(self.env.channels)))
        #return decision
        ##return tuple(a if s else None for s,a in zip(sensor_decision, decision))
