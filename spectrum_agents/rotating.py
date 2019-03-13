from spectrum_agents import Agent
from collections import deque

class Rotating(Agent):
    """
    Rotating Agent.
    Starts in an initial channel (provided or random) and then
    rotates through available channels in order each epoch (time step).
    Rotating agent initialization handled by inherited functionality from Agent.
    """
    def __init__(self, agent_id, N, seed=None, start=None):
        super().__init__(agent_id, N, seed=seed, start=start)
        try:
            self.a = deque(self.a)
        except:
            pass

    def _decide(self,o=None,r=None):
        if isinstance(self.a, deque):
            self.a.rotate()
            return tuple(self.a)
        else:
            return self.a

    def _learn(self,o=None,r=None):
        pass
