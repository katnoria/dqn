import sys
from collections import deque

import numpy as np
from unityagents import UnityEnvironment
from agents import DQNAgent

import torch

# DEFAULTS
NUM_EPISODES = 100      # Number of episodes
DEFAULT_EPS = 0.01      # default epsilon


class Player():
    def __init__(self):
        """Player implementation of dqn and random agents"""
        self.env = UnityEnvironment(file_name="../env/Banana_Linux_NoVis/Banana.x86_64")
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        # reset the environment
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        # number of actions
        self.action_size = brain.vector_action_space_size
        # examine the state space 
        state = env_info.vector_observations[0]
        state_size = len(state)

        self.agent = DQNAgent(state_size, self.action_size, seed=0)
        self.agent.local_network.load_state_dict(torch.load('../saved_models/dqn_banana_best.pth'))

    def play(self): 
        """Play using best dqn agent"""   
        scores = []
        scores_window = deque(maxlen=10)
        best_score = -np.inf
        eps = DEFAULT_EPS
        for i in range(NUM_EPISODES):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            while True:
                action = self.agent.act(state, eps)
                env_info = self.env.step(action)[self.brain_name]        
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                score += reward
                state = next_state
                if done:
                    break
                    
            scores_window.append(score)
            scores.append(score)        

            if i % 10 == 0:
                print('\rProgress: {}/{}, avg score: {:.2f}'.format(i, NUM_EPISODES, np.mean(scores_window)), end="")
                sys.stdout.flush()

        return scores, best_score


    def play_random(self):
        """Play by choosing random actions"""
        scores = []
        scores_window = deque(maxlen=10)
        best_score = -np.inf
        eps = DEFAULT_EPS
        for i in range(NUM_EPISODES):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            while True:
                action =  np.random.randint(self.action_size)
                env_info = self.env.step(action)[self.brain_name]        
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                score += reward
                state = next_state
                if done:
                    break
                    
            scores_window.append(score)
            scores.append(score)        

            if i % 10 == 0:
                print('\rProgress: {}/{}, avg score: {:.2f}'.format(i, NUM_EPISODES, np.mean(scores_window)), end="")
                sys.stdout.flush()

        return scores, best_score        
    