import numpy as np
from gym_spectrum.envs import SpectrumEnv
import pdb

class Agent(object):
    """
    Agent base class.
    Subclasses must implement decide and learn methods.
    Subcalsses must decide which observations are relevant.
    """
    def __init__(self, agent_id, env, sensors=None, seed=None, start=None):
        self.id = agent_id
        self.env = env
        self.state = tuple(None for _ in env.channels)
        self.action = tuple(None for _ in env.channels)
        self.sensors = sensors
        self.start = start
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def step(self, observation, reward):
        if self.start is not None:
            self.action = self.start
            self.start = None
        else:
            self.learn(observation, reward)
            self.action = self.decide(observation, reward)
        self.state = observation
        return self.action

    def partial_observation(self, observation):
        partial = np.array([
                None if (a==0) else o
                for a,o in zip(self.action, observation)])
        # randomly fill 'None' slots with observations
        # until all sensors are used
        none_arg = [i for i,x in enumerate(partial) if x is None]
        used_sensors = len(partial) - len(none_arg)
        rem_sensors = self.sensors - used_sensors
        rnd_observation = self.rng.choice(none_arg, rem_sensors, replace=False)
        if rnd_observation.size != 0:
            partial[rnd_observation] = [observation[x] for x in rnd_observation]
        return tuple(partial)
        #return tuple(x.item() for x in partial)

    def decide(self, observation, reward):
        raise NotImplementedError

    def learn(self, observation, reward):
        raise NotImplementedError
