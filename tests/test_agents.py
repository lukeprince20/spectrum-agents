from gym_spectrum.envs import ChannelEnv, SpectrumEnv
from spectrum_agents import Random, Q, Incumbent, Rotating, Bandit, Noop

def run_test(agent, env):
    print("Running Test for Agent: {}\n".format(agent.id))
    _, o = env.reset()
    r = None
    done = False
    while not done:
        a = agent.step(o=o, r=r)
        (o, r, done, _) = env.step(a)
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
    print("\n\n")

# Create environments
channel_env = ChannelEnv(alpha=0.2, beta=0.7, epochs=50)
spectrum_env = SpectrumEnv(channels=2, alphas=[0.2, 0.7], betas=[0.2, 0.7], epochs=50)

# Create agents
agent_noop = Noop('Noop', 2, start=0) # id: Noop, channels: 2, starting action: 0
agent_incumbent = Incumbent('Incumbent', 2, start=0) # id: Incumbent, channels: 2, starting action: 0
agent_rotating = Rotating('Rotating', 2, start=0) # id: Rotating, channels: 2, starting action: 0
agent_random = Random('Random', 2, start=0) # id: Random, channels: 2, starting action: 0
agent_bandit = Bandit('Bandit', 2, start=0) # id: Bandit, channels: 2, starting action: 0
agent_q = Q('Q', 2, start=0) # id: Q, channels: 2, starting action: 0

# Test agents
run_test(agent_noop, channel_env)
run_test(agent_incumbent, channel_env)
run_test(agent_rotating, channel_env)
run_test(agent_random, channel_env)
run_test(agent_bandit, channel_env)
run_test(agent_q, channel_env)


