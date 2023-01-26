import numpy as np
from collections import deque
import random

# Ornstein-Ulhenbeck Process
#Base from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.15, min_sigma=0, decay_period=500):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        if hasattr(action_space, "shape"):
            self.action_dim   = action_space.shape[0]
        else:
            self.action_dim   = action_space
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.step = 0
        
    def get_noise(self):
        if self.step > self.decay_period:
            print('Warning: current step > decay_period in OUNoise noise generator.')
        
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.step / self.decay_period)
        self.step += 1
        return self.state
    
    
class metaNoise:
    def __init__(self, noise_list):
        self.noise_list = noise_list
    def reset(self):
        for n in self.noise_list:
            n.reset()
    def get_noise(self):
        noise = []
        for n in self.noise_list:
            noise.append(n.get_noise())
        n_len = np.array([len(x) for x in noise])
        max_len = np.max(n_len)
        out_noise = np.zeros(max_len)
        for n in noise:
            out_noise += n
        return out_noise
    
class noNoise:
    def __init__(self):
        pass
    def reset(self):
        pass
    def get_noise(self):
        return [0]


# Base from https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, done):
        experience = (state, action, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    


class Memory_MF:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, mf_actions, obs_others, done):
        experience = (state, action, np.array([reward]), mf_actions, obs_others, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        mf_actions_batch = []
        obs_others_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, mf_actions, obs_others, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            mf_actions_batch.append(mf_actions)
            obs_others_batch.append(obs_others)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, mf_actions_batch, obs_others_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
    

class meta_memory_mf:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mem_type = []
        self.mem_counter = []
        self.memories = []
        
    def push(self, state, action, reward, mf_actions, obs_others, done):
        n_others = len(obs_others)
        if not n_others in self.mem_type:
            self.mem_type.append(n_others)
            self.mem_counter.append(1)
            self.memories.append(Memory_MF(self.max_size))
            
            memory_idx = self.mem_type.index(n_others)
            self.memories[memory_idx].push(state, action, reward, mf_actions, obs_others, done)
        else:
            memory_idx = self.mem_type.index(n_others)
            self.mem_counter[memory_idx] += 1
            self.memories[memory_idx].push(state, action, reward, mf_actions, obs_others, done)
            
    def sample(self, batch_size):
                
        avail_mem_idx = np.where(np.array(self.mem_counter) > batch_size)[0]
        avail_count = np.array([self.mem_counter[i] for i in avail_mem_idx])
        
        prob_vector = np.cumsum(avail_count) / np.sum(avail_count)
        idx = np.where(np.random.rand() <= prob_vector)[0][0]
        
        return self.memories[avail_mem_idx[idx]].sample(batch_size)
        
    def __len__(self):
        return max(self.mem_counter)