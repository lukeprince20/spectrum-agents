import itertools
from functools import wraps
import random
from math import floor

# decorator
def random_buffer(func):
    """
    decorator which returns an additional random delay
    between [0,1) along with the result of the orginal
    func call.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        delay = random.random()
        return result, delay
    return wrapper

# decorator
def onehot(func):
    """
    decorator which performs an exponential operation,
    raising 2 to the power of the returned result from the
    original func call. It is expected that the original
    result is a scalar int value.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            onehot = 2 ** result
            return onehot
        except ArithmeticError as e:
            errStr = "expected 'result' to be of type int\n" + \
                "but was of type {} instead.".format(type(result))
            print(errStr)
    return wrapper

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

        assert( seed is None or isinstance(seed,int) ), "seed must be integer or of NoneType"
        self._seed = seed

        assert( start is None or (0 <= floor(start) < self.N) ), "start must be a valid decision value or of NoneType"
        self._start = floor(start) if start is not None else None

    def step(self, o=None, r=None, info=None):
        self.decide(o=o, r=r) # updates self.a
        self.learn(o=o, r=r)
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

