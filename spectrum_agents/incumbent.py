from spectrum_agents import Agent

class Incumbent(Agent):
    """
    Incumbent agent starts in an initial channel
    (chosen or random) and stays there forever.
    Initial channel assignment handled in functionality
    inherited from Agent.
    Basically, this agent does nothing...
    """
    def _decide(self,o=None,r=None):
        return self.a

    def _learn(self,o=None,r=None):
        pass
