#!/usr/bin/env python
import numpy as np
import csv
import pandas as pd
from gym_spectrum.envs import SpectrumEnv
from spectrum_agents import Random, Genie, BeliefGenie, ReplicatedQ
from matplotlib import pyplot as plt
import pdb

def run_simulation(agent, env):
    # initialize
    _, observation = env.reset()
    reward = [0.0] * len(observation)
    done = False
    epoch_dict = {
        'return': np.zeros(env.maxEpochs),
        'reward': np.zeros(env.maxEpochs),
        'penalty': np.zeros(env.maxEpochs),
        'null': np.zeros(env.maxEpochs)}

    # simulation loop
    for i in range(env.maxEpochs):
        if not done:
            action = agent.step(observation=observation, reward=reward)
            (observation, reward, done, _) = env.step(action, mode='access')
            epoch_dict['return'][i] = sum(reward)
            epoch_dict['reward'][i] = sum(x==1.0 for x in reward)
            epoch_dict['penalty'][i] = sum(x==-0.5 for x in reward)
            epoch_dict['null'][i] = sum(x==0.0 for x in reward)

            if sum(x is None for x in reward) > 0:
                pdb.set_trace()

    return epoch_dict

def random_sumrate(k,n):
    """normalized, expected sumrate for k random agents and n channels"""
    phi = (n - 1.0)/n
    return (1 - phi**k)

def worstcase_sumrate(n):
    """normalized, worst case sumrate with k agents incumbent on the same
    channel out of n channels"""
    return 1.0/n

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def createPlots():
    d = {}
    with open('files.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for line in csv_reader:
            for key, value in line.items():
                d.setdefault(key, []).append(value)

    d.update((
        ('sensor', [int(x) for x in d['sensor']]),
        ('mc', [int(x) for x in d['mc']])))

    # create indice lists (TODO automate and make generic)
    env3indices = list(i for i,v in enumerate(d['env']) if v == 'env3')
    env5indices = list(i for i,v in enumerate(d['env']) if v == 'env5')
    env8indices = list(i for i,v in enumerate(d['env']) if v == 'env8')
    env10indices = list(i for i,v in enumerate(d['env']) if v == 'env10')

    sensor1indices = list(i for i,v in enumerate(d['sensor']) if v == 1)
    sensor3indices = list(i for i,v in enumerate(d['sensor']) if v == 3)
    sensor5indices = list(i for i,v in enumerate(d['sensor']) if v == 5)
    sensor8indices = list(i for i,v in enumerate(d['sensor']) if v == 8)
    sensor10indices = list(i for i,v in enumerate(d['sensor']) if v == 10)

    env3sensor1indices = list(set(env3indices) & set(sensor1indices))
    env3sensor3indices = list(set(env3indices) & set(sensor3indices))
    env5sensor1indices = list(set(env5indices) & set(sensor1indices))
    env5sensor3indices = list(set(env5indices) & set(sensor3indices))
    env5sensor5indices = list(set(env5indices) & set(sensor5indices))
    env8sensor1indices = list(set(env8indices) & set(sensor1indices))
    env8sensor3indices = list(set(env8indices) & set(sensor3indices))
    env8sensor5indices = list(set(env8indices) & set(sensor5indices))
    env8sensor8indices = list(set(env8indices) & set(sensor8indices))
    env10sensor1indices = list(set(env10indices) & set(sensor1indices))
    env10sensor3indices = list(set(env10indices) & set(sensor3indices))
    env10sensor5indices = list(set(env10indices) & set(sensor5indices))
    env10sensor8indices = list(set(env10indices) & set(sensor8indices))
    env10sensor10indices = list(set(env10indices) & set(sensor10indices))


    # load DataFrames from files and create plots
    formatStr = "Spectrum Utility: {} Channel Subband, {} Sensors"
    createUtilityPlot(d, env3sensor1indices, 3, 1, formatStr.format(3, 1))
    createUtilityPlot(d, env3sensor3indices, 3, 3, formatStr.format(3, 3))
    createUtilityPlot(d, env5sensor1indices, 5, 1, formatStr.format(5, 1))
    createUtilityPlot(d, env5sensor3indices, 5, 3, formatStr.format(5, 3))
    createUtilityPlot(d, env5sensor5indices, 5, 5, formatStr.format(5, 5))
    createUtilityPlot(d, env8sensor1indices, 8, 1, formatStr.format(8, 1))
    createUtilityPlot(d, env8sensor3indices, 8, 3, formatStr.format(8, 3))
    createUtilityPlot(d, env8sensor5indices, 8, 5, formatStr.format(8, 5))
    createUtilityPlot(d, env8sensor8indices, 8, 8, formatStr.format(8, 8))
    createUtilityPlot(d, env10sensor1indices, 10, 1, formatStr.format(10, 1))
    createUtilityPlot(d, env10sensor3indices, 10, 3, formatStr.format(10, 3))
    createUtilityPlot(d, env10sensor5indices, 10, 5, formatStr.format(10, 5))
    createUtilityPlot(d, env10sensor8indices, 10, 8, formatStr.format(10, 8))
    createUtilityPlot(d, env10sensor10indices, 10, 10, formatStr.format(10, 10))

def createUtilityPlot(filedict, indices, channels, sensors, title):
    # create figure and axes
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim(-.25, 1.25)
    ax.set_title(title)
    ax.set_xlabel('Time Epochs')
    ax.set_ylabel('Average Utility')
    fig.axes.append(ax)

    ## plot random and worst case utility
    #random_utility = [random_sumrate(3,len(env.channels))] * 2
    #worstcase_utility = [worstcase_sumrate(len(env.channels))] * 2
    #plt.plot([1, maxEpochs], random_utility, 'k--', label='Random Utility')
    #plt.plot([1, maxEpochs], worstcase_utility, 'r--', label='Worst Case Utility')

    # read DataFrame from relevant files and plot utility
    for i in indices:
        df = pd.read_csv(filedict['file'][i])
        df['reward'] /= float(channels)
        df['reward'].plot(ax=ax, label=filedict['agent'][i])

    time = list(df.index)
    max_utility_inclusive = [float(sensors)/float(channels)] * 2
    plt.plot([time[0], time[-1]], max_utility_inclusive, 'k-')

    max_utility_exclusive = [(2./3.) * x for x in max_utility_inclusive]
    plt.plot([time[0], time[-1]], max_utility_exclusive, 'k--')

    # create legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    # save figure and close
    plt.savefig(title + '.png')
    plt.close()



if __name__ == "__main__":
    import os
    import datetime
    import pdb

    # create ouput directory
    cwd = os.getcwd()
    timeFormatSpec = '%Y%m%d_%H%M'
    currentTime = datetime.datetime.now().strftime(timeFormatSpec)
    filesep = os.sep
    path = filesep.join([cwd, 'outputData_{}'.format(currentTime)])
    os.mkdir(path)
    os.chdir(path)

    # create csv file object and writer
    csv_file = open('files.csv', 'w');
    csv_writer = csv.writer(csv_file)
    header = ['file', 'env', 'sensor', 'agent', 'mc']
    csv_writer.writerow(header)

    # Create RNG
    rng = np.random.RandomState()

    # create environments
    env3 = SpectrumEnv(alphas=[0.1] * 3, betas=[0.2] * 3, epochs=40)
    env5 = SpectrumEnv(alphas=[0.1] * 5, betas=[0.2] * 5, epochs=40)
    env8 = SpectrumEnv(alphas=[0.1] * 8, betas=[0.2] * 8, epochs=40)
    env10 = SpectrumEnv(alphas=[0.1] * 10, betas=[0.2] * 10, epochs=40)
    envs = (env3, env5, env8, env10)
    envNames = ('env3', 'env5', 'env8', 'env10')

    for envName, env in zip(envNames, envs):

        # create sensors
        if envName == 'env3':
            sensors = (1, 3)
        elif envName == 'env5':
            sensors = (1, 3, 5)
        elif envName == 'env8':
            sensors = (1, 3, 5, 8)
        elif envName == 'env10':
            sensors = (1, 3, 5, 8, 10)

        for sensor in sensors:
            def startFcn():
                i = rng.choice(len(env.channels), rng.randint(0, sensor+1), replace=False)
                start = np.zeros(len(env.channels), np.int64)
                start[i] = 1
                return tuple(x.item() for x in start)

            # create agents
            random = Random('Random', env, seed=rng.randint(0, 2**32), start=startFcn(), sensors=sensor)
            genie = Genie('Genie', env, seed=rng.randint(0, 2**32), start=startFcn(), sensors=sensor)
            belief_genie = BeliefGenie('BeliefGenie', env, seed=rng.randint(0, 2**32), start=startFcn(), sensors=sensor)
            replicated_q = ReplicatedQ('ReplicatedQ', env, seed=rng.randint(0, 2**32), start=startFcn(), sensors=sensor)
            agents = (random, genie, belief_genie, replicated_q)

            # run MC simulations
            for agent in agents:
                numSims = 200
                df = pd.DataFrame({
                    'return': np.zeros(env.maxEpochs),
                    'reward': np.zeros(env.maxEpochs),
                    'penalty': np.zeros(env.maxEpochs),
                    'null': np.zeros(env.maxEpochs)})

                for s in range(numSims):
                    run_dict = run_simulation(agent, env)
                    df += pd.DataFrame(run_dict)

                df /= float(numSims)
                df.index = df.index + 1

                # write to DataFrame csv file
                fields = [envName, str(sensor), agent.id, str(numSims)]
                filename = "_".join(fields) + '.csv'
                df.to_csv(filename)

                # write to filename csv file
                fields.insert(0, filename)
                csv_writer.writerow(fields)

    # destroy csv file object
    csv_file.close()

    # create utility plots
    createPlots()

    # return to original directory
    os.chdir(cwd)


            ## create and save plots
            #createUtilityPlot(envName, env.maxEpochs, agents, utilityDict)
