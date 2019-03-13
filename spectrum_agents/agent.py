class Agent(object):
    """
    Agent base class. 
    Subclasses must implement _decide and _learn methods.
    """
    def __init__(self, agent_id,  N, seed=None, start=None):
        self.id = agent_id
        self.N = N # num channels
        self.s = None # state
        self.a = None # decision

        assert(seed is None or isinstance(seed,int)), "seed must be integer or None"
        self._seed = seed
        self._start = start

    def step(self, o=None, r=None, info=None):
        if self._start is not None:
            self.a = self._start
            self._start = None
        else:
            self.learn(o=o, r=r)
            self.decide(o=o, r=r) # updates self.a
        self.s = o # update state
        return self.a

    def decide(self, o=None, r=None):
        if self._start is not None:
            self.a = self._start
            self._start = None
        else:
            self.a = self._decide(o=o, r=r)

    def _decide(self, o=None, r=None):
        raise NotImplementedError

    def learn(self, o=None, r=None):
        self._learn(o=o, r=r)

    def _learn(self, o=None, r=None):
        raise NotImplementedError

