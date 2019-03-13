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
        try:
            m = tuple('no-op' if (x is None) else 'predict' for x in a)
        except:
            m = 'no-op' if (a is None) else 'predict'
        (o, r, done, _) = env.step(a, mode=m)
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
    print("\n\n")

# Create environment
env = SpectrumEnv(alphas=[0.1, 0.1], betas=[0.2, 0.2], epochs=50)

# Create agents
agent_noop = Noop('Noop', env, start=env.action_space.sample())
agent_incumbent = Incumbent('Incumbent', env, start=env.action_space.sample())
agent_random = Random('Random', env, start=env.action_space.sample())
agent_genie = Genie('Genie', env, start=env.action_space.sample())
agent_belief_genie = BeliefGenie('BeliefGenie', env, start=env.action_space.sample())
agent_replicated_q = ReplicatedQ('ReplicatedQ', env, start=env.action_space.sample())

# Run tests
run_test(agent_noop, env)
run_test(agent_incumbent, env)
run_test(agent_random, env)
run_test(agent_genie, env)
run_test(agent_belief_genie, env)
run_test(agent_replicated_q, env)
