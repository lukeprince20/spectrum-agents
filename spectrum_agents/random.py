from spectrum_agents import Agent
import random

class Random(Agent):
    """
    Random Agent inheriting from Agent base class.
    Picks channels at random and learns nothing.
    """
    def _decide(self,o=None,r=None):
        if hasattr(o, "__iter__"):
            return tuple(random.randint(0, 1) for _ in o)
        else:
            return random.randint(0, 1)

    def _learn(self,o=None,r=None):
        pass
