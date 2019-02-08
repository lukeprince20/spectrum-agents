from spectrum_agents import Agent

class Rotating(Agent):
    """
    Rotating Agent.
    Starts in an initial channel (provided or random) and then
    rotates through available channels in order each epoch (time step).
    Rotating agent initialization handled by inherited functionality from Agent.
    """
    def _decide(self,o=None,r=None):
        return (self.a + 1) % self.N

    def _learn(self,o=None,r=None):
        pass
