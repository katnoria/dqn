import os
import random 
import numpy as np
from collections import deque, namedtuple, defaultdict
from model import LinearModel

import torch
import torch.nn.functional as F
import torch.optim as optim

import logging

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s [%(threadName)12.12s] [%(levelname)-5.5s] %(message)s")
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler('logs/agent.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(sh)
logger.addHandler(fh)


# Default parameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQNAgent():
    """Implementation of DQN"""

    def __init__(self, state_size, action_size, seed, opts={}):
        """Initialise the agent

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): Random seed
            opts (dict): optional settings
                BUFFER_SIZE (long): Max size for replay buffer
                BATCH_SIZE (int): Sample size of experiences from replay buffer
                GAMMA (float): discount factor
                TAU (float): soft update of target parameters
                LR (float): optimizer learning rate
                UPDATE_EVERY (int): how ofter to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.local_network = LinearModel(state_size, action_size, seed).to(device)
        self.fixed_network = LinearModel(state_size, action_size, seed).to(device)

        # Overwrite the default parameters
        self.buffer_size  = opts['BUFFER_SIZE'] if 'BUFFER_SIZE' in opts else BUFFER_SIZE
        self.batch_size = opts['BATCH_SIZE'] if 'BATCH_SIZE' in opts else BATCH_SIZE        
        self.gamma = opts['GAMMA'] if 'GAMMA' in opts else GAMMA
        self.tau = opts['TAU'] if 'TAU' in opts else TAU
        self.lr = opts['LR'] if 'LR' in opts else LR
        self.update_every = opts['UPDATE_EVERY'] if 'UPDATE_EVERY' in opts else UPDATE_EVERY
        self.optim = optim.Adam(self.local_network.parameters(), lr=self.lr)

        # Initialize replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.history = defaultdict(list)

    def act(self, state, eps):
        """Returns the action for specified state

        Params
        ======
            state (array_like): environment state
            eps (float): epsilon, to use in greedy-policy
        """
        if random.random() < eps:
            return random.choice(range(self.action_size))
        else:
            # convert the state to tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # change network into evaluation mode with no gradients
            self.local_network.eval()
            with torch.no_grad():
                action_values = self.local_network(state)
            # change network back to training mode
            self.local_network.train()
            return np.argmax(action_values.cpu().data.numpy())


    def step(self, state, action, reward, next_state, done):
        """Collects experience and learns from experience

        Params
        ======
            state (array_like): environment state (S)
            action (int): action taken on state (A)
            reward (float): reward (R) received by taking action A in state S
            next_state (array_like): environment state (S') received after taking action A in state S
            done (boolean): whether the episode ended after taking action A in state S
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        

    def learn(self, experiences, gamma):
        """Use experience to learn from it

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # logger.debug('learn gamma: {}'.format(gamma))
        states, actions, rewards, next_states, dones = experiences
        Q_next = self.fixed_network(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (gamma * Q_next * (1 - dones))

        Q_estimates = self.local_network(states).gather(1, actions)

        loss = F.mse_loss(Q_estimates, Q_target)
        self.history['loss'].append(loss.item())
        # logger.debug('Loss: {}'.format(loss.item()))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update fixed network
        self.update_fixed_network(self.local_network, self.fixed_network, self.tau)
        

    def update_fixed_network(self, local_model, target_model, tau):
        """Updates fixed target network weights using following:
        target = tau * target_model weights + (1 - tau) * local_model weights

        Params
        ======
            local_model (Pytorch model): source model to copy weights from
            target_model (Pytorch model): target model to copy weights to
            tau (float): decide how much weight to apply when updating weights                
        """
        for target_weights, local_weights in zip(target_model.parameters(), local_model.parameters()):
            target_weights.data.copy_(tau * target_weights + (1 - tau) * local_weights)

    def save(self, filename):
        """Save local model parameters
        
        Params
        ======
            filename (string): filename
        """
        torch.save(self.local_network.state_dict(), filename)

class ReplayBuffer():

    def __init__(self, buffer_size, batch_size, seed):
        """Initialise the replay buffer

        Params
        ======
            buffer_size (int): maximum size of memory
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)        

    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Sample experiences from memory"""        
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert to pytorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Returns the current size of internal memory"""
        return len(self.memory)
