from spectrum_agents import Agent
import random

class Random(Agent):
    """
    Random Agent inheriting from Agent base class.
    Picks channels at random and learns nothing.
    """
    def _decide(self,o=None,r=None):
        return random.randint(0, self.N-1) # {0,...,N-1}

    def _learn(self,o=None,r=None):
        pass
