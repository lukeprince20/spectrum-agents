import numpy as np
from gym_spectrum.envs import SpectrumEnv
from spectrum_agents import Random, Incumbent, Rotating, Noop
from spectrum_agents import Genie, BeliefGenie, ReplicatedQ

def run_test(agent, env):
    print("Running Test for Agent: {}\n".format(agent.id))
    _, o = env.reset()
    r = None
    done = False
    while not done:
        a = agent.step(observation=o, reward=r)
        m = 'access'
        (o, r, done, _) = env.step(a,mode=m)
        print("Action Taken: ", a, "; ", env.render(mode="string"), ";Observation: ", agent.partial_observation(o), "; Reward: ", r)
    print("\n\n")

# Create RNG
rng = np.random.RandomState()

# Create environment
env = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=50)

# Create agents
agent_noop = Noop('Noop', env, seed=rng.randint(0, 2**32), start=env.action_space.sample())
agent_incumbent = Incumbent('Incumbent', env, seed=rng.randint(0, 2**32), start=env.action_space.sample())
agent_random = Random('Random', env, seed=rng.randint(0, 2**32), start=env.action_space.sample())
agent_genie = Genie('Genie', env, seed=rng.randint(0, 2**32), start=env.action_space.sample())
agent_belief_genie = BeliefGenie('BeliefGenie', env, seed=rng.randint(0, 2**32), start=env.action_space.sample(), sensors=2)
agent_replicated_q = ReplicatedQ('ReplicatedQ', env, seed=rng.randint(0, 2**32), start=env.action_space.sample(), sensors=2)

# Run tests
run_test(agent_noop, env)
run_test(agent_incumbent, env)
run_test(agent_random, env)
run_test(agent_genie, env)
run_test(agent_belief_genie, env)
run_test(agent_replicated_q, env)
