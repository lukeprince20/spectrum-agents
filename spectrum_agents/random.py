from spectrum_agents import Agent

class Random(Agent):
    """
    Random Agent inheriting from Agent base class.
    Picks channels at random and learns nothing.
    """
    def decide(self, observation, reward):
        if hasattr(observation, "__iter__"):
            return tuple(self.rng.randint(0, 2) for _ in observation)
        else:
            return self.rng.randint(0, 2)

    def learn(self, observation, reward):
        pass
