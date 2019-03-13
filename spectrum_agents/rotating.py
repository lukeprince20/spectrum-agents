from spectrum_agents import Agent
from collections import deque

class Rotating(Agent):
    """
    Rotating Agent.
    Starts in an initial channel (provided or random) and then
    rotates through available channels in order each epoch (time step).
    Rotating agent initialization handled by inherited functionality from Agent.
    """
    def __init__(self, agent_id, env, seed=None, start=None):
        super().__init__(agent_id, env, sensors=len(env.channels), seed=seed, start=start)
        try:
            self.action = deque(self.action)
        except:
            pass

    def decide(self, observation, reward):
        if isinstance(self.action, deque):
            self.action.rotate()
            return tuple(self.action)
        else:
            return self.action

    def learn(self, observation, reward):
        pass
