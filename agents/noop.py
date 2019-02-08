from agents import Agent

class Noop(Agent):
    """
    Similar to Incumbent, but always returns no operation (no-op)
    """
    def _decide(self,o=None,r=None):
        return None

    def _learn(self,o=None,r=None):
        pass
