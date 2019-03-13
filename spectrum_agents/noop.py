from spectrum_agents import Agent

class Noop(Agent):
    """
    Similar to Incumbent, but always returns no operation (no-op)
    """
    def decide(self, observation, reward):
        if hasattr(observation, "__iter__"):
            return tuple(None for _ in observation)
        else:
            return None

    def learn(self, observation, reward):
        pass
