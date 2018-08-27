[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation

### Introduction

In this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and is considered solved when our agent is able to get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Run `python setup.py` to install the dependencies

2. Download the unity ml environment from one of the links below and extract it under **env** folder.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    

### Instructions

The repository is organised as following:
- **`env`** - stores all the unity ml environments
- **`notebooks`** - stores the notebooks for environment exploration, training agents and hyperparameter tuning.
- **`saved_models`** - stores saved network weights
- **`src`** - includes the code for agent and neural networks. agents.py contains all the agents, model.py contains neural network implementation and player.py allows us to run the trained model.

Use notebooks/Trainer.ipynb to see how the agent was trained.

The file **Report.ipynb** includes the information of how the agent was trained, its performance against a player that chooses actions randomly and future ideas. 

*Report.pdf was generated from the Report notebook. It is best to view Report.ipynb and use pdf only in cases where Reports notebook cannot be opened/rendered*

**Note:** *The training and testing was performed on local Ubuntu 16.04.*
