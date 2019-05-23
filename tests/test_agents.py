import numpy as np
from gym_spectrum.envs import SpectrumEnv
from spectrum_agents import Random, Incumbent, Rotating, Noop
from spectrum_agents import Genie, BeliefGenie, ReplicatedQ

def run_test(agent, env):
    print("Running Test for Agent: {}\n".format(agent.id))
    _, o = env.reset()
    #r = None
    r = [0.0] * len(o)
    done = False
    while not done:
        a = agent.step(observation=o, reward=r)
        m = 'access'
        (o, r, done, _) = env.step(a,mode=m)
        print("Action Taken: ", a, "; ", env.render(mode="string"), ";Observation: ", agent.partial_observation(o), "; Reward: ", r)
    print("\n\n")

# Create RNG and common seed
rng = np.random.RandomState()
seed = rng.randint(0, 2**32)

# Create environment and common starting actions
env = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=50, seed=seed)
agent_random = Random('Random', env, seed=seed, start=env.action_space.sample(), sensors=4)
run_test(agent_random, env)

env = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=50, seed=seed)
agent_genie = Genie('Genie', env, seed=seed, start=env.action_space.sample(), sensors=4)
run_test(agent_genie, env)

env = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=50, seed=seed)
agent_belief_genie = BeliefGenie('BeliefGenie', env, seed=seed, start=env.action_space.sample(), sensors=4)
run_test(agent_belief_genie, env)

env = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=50, seed=seed)
agent_replicated_q = ReplicatedQ('ReplicatedQ', env, seed=seed, start=env.action_space.sample(), sensors=4)
run_test(agent_replicated_q, env)
