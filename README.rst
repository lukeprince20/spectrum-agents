**Status:** Development (expect API changes and major updates)

Spectrum-Agents
***************

Reinforcement learning (RL) agents intended for use with gym-spectrum OpenAI Gym environments.

Overview
========
Todo

Dependencies
============
Written and tested on Python 3.6. Requires OpenAI Gym, Numpy, and Matplotlib.

Installation
============
It is recommend to create a venv.

.. code:: shell

    python3 -m venv spectrum
    source ./spectrum/bin/activate

To deactivate the venv, simply run the following in the shell:

.. code:: shell

    deactivate

Direct from Git
---------------
You can clone the repo and install the module directly from your local Git clone using a pip editable installation.

.. code:: shell

    git clone https://github.com/lukeprince20/spectrum-agents.git
    cd spectrum-agents
    pip install -e .

Supported Environments
======================
The environments the spectrum agents are written to operate on are maintained in a separate repo: https://github.com/lukeprince20/gym-spectrum.

Agents
======
Currently agent support is varied per environment.

ChannelEnv Only
---------------
- Bandit
- Q-Learning

SpectrumEnv Only
----------------
- Genie
- Belief State Genie
- Replicated Q-Learning

Both
----
- No-op
- Incumbent
- Rotating
- Random


