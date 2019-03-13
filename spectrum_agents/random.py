from spectrum_agents import Agent
import random

class Random(Agent):
    """
    Random Agent inheriting from Agent base class.
    Picks channels at random and learns nothing.
    """
    def decide(self, observation, reward):
        if hasattr(observation, "__iter__"):
            return tuple(random.randint(0, 1) for _ in observation)
        else:
            return random.randint(0, 1)

    def learn(self, observation, reward):
        pass
