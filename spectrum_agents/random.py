from spectrum_agents import Agent
import numpy as np

class Random(Agent):
    """
    Random Agent inheriting from Agent base class.
    Picks channels at random and learns nothing.
    """
    def decide(self, observation, reward):
        #if hasattr(observation, "__iter__"):
        #    return tuple(self.rng.randint(0, 2) for _ in observation)
        #else:
        #    return self.rng.randint(0, 2)
        random_decision = np.zeros(len(observation), np.int64)
        sensor_index = self.rng.choice(len(observation), self.sensors,
                replace=False)
        random_decision[sensor_index] = self.rng.choice([0, 1], self.sensors)
        return random_decision

    def learn(self, observation, reward):
        pass
