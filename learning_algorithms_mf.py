'''
Risk-Aware Continuous Control with Neural Contextual Bandits
	  
  File:     learning_algorithms_mf.py 
  Authors: Jose A. Ayala-Romero (jose.ayala@neclab.eu) 
			Andres Garcia-Saavedra (andres.garcia.saavedra@neclab.eu) 
			Xavier Costa-Perez (xavier.costa@neclab.eu)


NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
'''


import numpy as np
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import os
import copy

from utils_learning import noNoise
# from utils_learning import Memory_MF as Memory
from utils_learning import meta_memory_mf as Memory
from models import MLP, ConvNet, BuildConvActorCritic


def process_data_mf(state, action, reward, action_mf_dim):
    obs_proc = []
    actions_proc = []
    rewards_proc = []
    mf_actions_proc = []
    obs_others_mf_comp = []
    
    aggregated_state = np.sum(state, axis=0)
    obs_aux_1 = []
    
    for i in range(len(state)):
        obs_minus_i = aggregated_state - state[i]
        obs_i_mf = np.concatenate((state[i], obs_minus_i), axis=0)
        obs_proc.append(obs_i_mf)
        obs_aux_1.append(obs_i_mf)
        actions_proc.append(action[i])


        mask = np.ones(len(action), dtype=bool)
        mask[i] = False
        action_vector_aux = action[mask]
        
        hist_i, _ = np.histogram(action_vector_aux, bins=action_mf_dim, range=(0,1), density=False)
        norm_value = len(action) - 1 if len(action)>0 else 1 
        hist_i_norm = hist_i / norm_value
        mf_actions_proc.append(hist_i_norm)
        
        rewards_proc.append(reward)
        
    # obs_aux_2 = []
    for i in range(len(obs_aux_1)):
        obs_i = copy.deepcopy(obs_aux_1)
        del obs_i[i]
        obs_others_mf_comp.append(obs_i)
            
    return obs_proc, actions_proc, rewards_proc, mf_actions_proc, obs_others_mf_comp



class LearningAlgorithmBaseMF:
    def __init__(self, env, noise, batch_size, save_model, updates_per_step, device):
        self.env = env
        if noise is not None:
            self.noise = noise
        else:
            self.noise = noNoise()
        self.batch_size = batch_size
        self.save = save_model
        self.updates_per_step = updates_per_step
        self.dim_states = self.env.observation_space.shape[0]
        self.dim_actions = self.env.action_space.shape[0]
        self.action_mf_dim = self.env.action_mf_dim
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.save_path = self.env.path
        self.device = device
        self.n_bss = -1
        self.process_data_class = self.env.process_data_class
        self.last_state = None
        self.last_info = None
        self.noise.reset()

    def denormalize_action(self, action):
        
        action = np.minimum(action, 1)
        action = np.maximum(action, 0)
        denorm_act = action * (self.action_high - self.action_low) + self.action_low
        return denorm_act
    
    def get_action(self, state):
        state = state[0]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action
            

    def run(self, n_iters, evaluation=0):
        
        for mod in self.learning_modules:
            if evaluation == 0:
                mod.train()
            else:
                mod.eval()
        self.n_bss = self.env.n_bss

        if self.last_state is None:
            initial_action = np.array([self.env.action_space.sample() for _ in range(self.n_bss)])
            state, _, _, info = self.env.step(initial_action)  
        else:
            state = self.last_state
            info = self.last_info

        for step in range(n_iters-1):
            if (step+1) % 1 == 0:
                print(f'Step {step+1} / {n_iters}')
                
            action_env = np.ones((self.n_bss,1)) * np.nan
            action = []
            
            state_agregated = np.sum(state, axis=0)
            for i, obs_i in zip(info['active_bss'], state):
                
                obs_minus_i = state_agregated - obs_i
                obs_i_mf = np.concatenate((obs_i, obs_minus_i), axis=0)
                obs_i_mf = np.expand_dims(obs_i_mf, axis=0)
                action_i = self.get_action(obs_i_mf)
                # print(f'obs_{i}: {obs_i_mf}; action_{i}: {action_i}')
                action_env[i] = action_i
                action.append(action_i)
            action = np.array(action)
            
            if evaluation == 0:
                noise = self.noise.get_noise()
                if len(noise) == len(action_env):
                    action += noise[~np.isnan(action_env.reshape(1,-1)[0])].reshape(-1,1)
                    action_env += noise.reshape(-1,1)
                elif len(noise) == 1:
                    action += noise[0]
                    action_env += noise[0]
                else:
                    raise Exception(f'Generated noise with length {len(noise)} and action with length {len(action)}')

            denorm_action = self.denormalize_action(action_env)
            
            new_state, reward, done, info = self.env.step(denorm_action) 
            # print(f'action: {denorm_action}; reward: {reward}')
            
            if evaluation == 0:
                # # if we save the data without processing
                # self.memory.push(state, action, reward, done)

                mf_data = process_data_mf(state, action, reward, self.action_mf_dim)
                states, actions, rewards, mf_actions, obs_others = mf_data
                
                for i in range(len(states)):
                    self.memory.push(states[i], actions[i], rewards[i], mf_actions[i], obs_others[i], done)

                if len(self.memory) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        self.update()
            state = new_state
            
            
            if done:
                break
        self.last_state = state
        self.last_info = info
        self.env.close()
        
        if self.save:
            self.save_models()

    def save_models(self):
        torch.save(self.actor.state_dict(), os.path.join(self.save_path, 'actor_model'))
        torch.save(self.critic.state_dict(), os.path.join(self.save_path, 'critic_model'))




class ECORAN_MF(LearningAlgorithmBaseMF):
    def __init__(self, env, noise, batch_size, actor_kwargs, critic_kwargs, convnet_kwargs=None,
                 actor_learning_rate=1e-4, critic_learning_rate=1e-3, max_memory_size=2000, save_model=0,
                 updates_per_step=1, device='cpu'):
        super().__init__(env, noise, batch_size, save_model, updates_per_step, device)
        
        if convnet_kwargs is None:
            dim_mf_state = self.dim_states
            actor_kwargs.update({'in_features' : self.dim_states + dim_mf_state, 'out_features' : self.dim_actions, 'device' : device})
            self.actor = MLP(**actor_kwargs)
            
            critic_input_size = self.dim_states + dim_mf_state + self.dim_actions + self.action_mf_dim
            critic_kwargs.update({'in_features' : critic_input_size, 'out_features' : 1, 'device' : device})
            self.critic = MLP(**critic_kwargs)
        else:
            extended_context_dim = [2*self.dim_states, self.dim_states, self.dim_states]
            convnet_kwargs.update({'device' : device})
            conv_net_actor = ConvNet(**convnet_kwargs)
            actor_kwargs.update({'out_features' : self.dim_actions, 'device' : device})
            self.actor = BuildConvActorCritic(conv_net_actor, actor_kwargs, extended_context_dim)
            
            conv_net_critic = ConvNet(**convnet_kwargs)
            critic_kwargs.update({'out_features' : 1, 'device' : device})
            self.critic = BuildConvActorCritic(conv_net_critic, critic_kwargs, extended_context_dim, self.dim_actions, self.action_mf_dim)
        
        self.learning_modules = [self.actor, self.critic]
        print(self.actor)
        print(self.critic)
                
        # Training
        self.memory = Memory(max_memory_size)  
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    
    def update(self):
        states, actions, rewards, mf_actions, obs_others, _ = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        mf_actions = torch.FloatTensor(np.array(mf_actions)).to(self.device)
        obs_others = torch.FloatTensor(np.array(obs_others)).to(self.device)
        
        # Critic loss
        Qvals = self.critic.forward(states, actions, mf_actions)
        critic_loss = self.critic_criterion(Qvals, rewards)
        
        # Computation of the mean field action
        actor_mf_actions_hist = []
        for i in range(self.batch_size):
            actor_mf_actions_i = self.actor.forward(obs_others[i])
            h = torch.histc(actor_mf_actions_i, self.action_mf_dim, min=0, max=1, out=None)
            norm_value = len(actor_mf_actions_i) if len(actor_mf_actions_i)>0 else 1
            h = torch.div(h ,norm_value)
            actor_mf_actions_hist.append(h)
        actor_mf_actions_hist = torch.stack(actor_mf_actions_hist)
        
        # policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() # maximize
        policy_loss = self.critic.forward(states, self.actor.forward(states), actor_mf_actions_hist).mean()  # minimize
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
