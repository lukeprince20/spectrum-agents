import numpy as np
from gym_spectrum.envs import ChannelEnv, SpectrumEnv
from spectrum_agents import Random, Q, Incumbent, Rotating, Bandit, Noop
from spectrum_agents import Genie, BeliefGenie, ReplicatedQ

def run_test(agent, env):
    print("Running Test for Agent: {}\n".format(agent.id))
    _, o = env.reset()
    r = None
    done = False
    while not done:
        a = agent.step(o=o, r=r)
        try:
            m = tuple('no-op' if (x is None) else 'predict' for x in a)
        except:
            m = 'no-op' if (a is None) else 'predict'
        (o, r, done, _) = env.step(a, mode=m)
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
    print("\n\n")

# Create environments
channel_env = ChannelEnv(alpha=0.1, beta=0.2, epochs=50)
spectrum_env = SpectrumEnv(alphas=[0.1, 0.1], betas=[0.2, 0.2], epochs=50)

for env in (channel_env, spectrum_env):
    agent_noop = Noop('Noop', 2, start=env.action_space.sample())
    agent_incumbent = Incumbent('Incumbent', 2, start=env.action_space.sample())
    agent_random = Random('Random', 2, start=env.action_space.sample())
    #agent_rotating = Rotating('Rotating', 2, start=env.action_space.sample())
    run_test(agent_noop, env)
    run_test(agent_incumbent, env)
    run_test(agent_random, env)
    #run_test(agent_rotating, env)

    if 0: #env == channel_env:
        agent_bandit = Bandit('Bandit', 2, start=env.action_space.sample())
        agent_q = Q('Q', 2, start=env.action_space.sample())
        run_test(agent_bandit, env)
        run_test(agent_q, env)

    if 0: #env == spectrum_env:
        agent_genie = Genie('Genie', spectrum_env, start=env.action_space.sample())
        agent_belief_genie = BeliefGenie('BeliefGenie', env, start=env.action_space.sample())
        agent_replicated_q = ReplicatedQ('ReplicatedQ', env, start=env.action_space.sample())
        run_test(agent_genie, env)
        run_test(agent_belief_genie, env)
        run_test(agent_replicated_q, env)

