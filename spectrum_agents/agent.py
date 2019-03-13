from gym_spectrum.envs import SpectrumEnv

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

    def step(self, observation, reward):
        if self.start is not None:
            self.action = self.start
            self.start = None
        else:
            self.learn(observation, reward)
            self.action = self.decide(observation, reward)
        self.state = observation
        return self.action

    def decide(self, observation, reward):
        raise NotImplementedError

    def learn(self, observation, reward):
        raise NotImplementedError
