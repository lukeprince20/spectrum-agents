from spectrum_agents import Agent

class Noop(Agent):
    """
    Similar to Incumbent, but always returns no operation (no-op)
    """
    def _decide(self,o=None,r=None):
        if hasattr(o, "__iter__"):
            return tuple(None for _ in o)
        else:
            return None

    def _learn(self,o=None,r=None):
        pass
