'''
Risk-Aware Continuous Control with Neural Contextual Bandits
	  
  File:     eval_learning_alg_mf.py 
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

import os
from datetime import datetime
from argparse import ArgumentParser
import json
import torch
import torch.nn as nn

# from env_wrapper import ORANHAEnv
from process_data_classes import ProcessDataTbsPdfMF, ProcessData3DPdfMF

from learning_algorithms_mf import ECORAN_MF
# from learning_algorithms_IL import ECORAN_IL
# from learning_algorithms_maddpg import ECORAN_MADDPG

from utils_learning import OUNoise
# from exploration import NoisyLinear


parser = ArgumentParser()
parser.add_argument("--policy", nargs='+', default=['tbsMultiThresholdPolicy'])
parser.add_argument("--input_dim", type=int, default=3)
parser.add_argument("--context_dim", type=int, default=5)
parser.add_argument("--delta", type=int, default=2)
parser.add_argument("--updates_per_step", type=int, default=5)
parser.add_argument("--training_epochs", type=int, default=1)
parser.add_argument('--bench_eval', action='store_true', default=False)
parser.add_argument('--noisyNet', action='store_true', default=False)
parser.add_argument("-T", "--timestamp", nargs='+', default=[''])
parser.add_argument("--bs_config_file", nargs='+', default=['bs_config_variable_paper.json'])
parser.add_argument("--bs_config_file_training", nargs='+', default=[''])
parser.add_argument("--ha_config_file", nargs='+', default=['ha_config_16.json'])
parser.add_argument("--max_n_users", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_iters", type=int, default=600)
parser.add_argument("--eval_iters", type=int, default=-1)
parser.add_argument("--n_context_av", type=int, default=1)
parser.add_argument("-t", "--tag", nargs='+', default=[''])
parser.add_argument('--ml_version', nargs='+', default=['mf_v1'])


args = parser.parse_args()
policy = args.policy[0]
input_dim = args.input_dim
context_dim = args.context_dim
delta = args.delta
updates_per_step = args.updates_per_step
training_epochs = args.training_epochs
bench_eval = args.bench_eval
noisyNet = args.noisyNet
timestamp = args.timestamp[0]
tag = args.tag[0]
ha_config_file = args.ha_config_file[0]
bs_config_file = args.bs_config_file[0]
bs_config_file_training = args.bs_config_file_training[0]
max_n_users = args.max_n_users
batch_size = args.batch_size
train_iters = args.train_iters
eval_iters = args.eval_iters
n_context_av = args.n_context_av
ml_version = args.ml_version[0]

if len(bs_config_file_training) == 0:
    bs_config_file_training = bs_config_file
    
if eval_iters == -1:
    eval_iters = train_iters
    
tag = '_'+tag if len(tag)>0 else tag

assert input_dim == 1 or input_dim == 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch device: {device}')

max_running_time = 200
restart = 0

if policy == 'tbsMultiThresholdPolicy':
    action_range = (32, 286976)
    action_dim = 1
else:
    raise Exception('Unknown policy.')
ha_assig_alg_str = policy
    
    
# action_range = (0, 3e-3)
# ha_assig_alg_str = 'timeThresholdPolicy'
# action_range = (32, 286976)
# ha_assig_alg_str = 'tbsThresholdPolicy'

kwargs = {'max_n_users' : max_n_users, 'delta' : delta, 'dim' : context_dim, 'action_mf_dim' : 5, 'n_context_av' : n_context_av}
# kwargs = {'max_n_users' : 6, 'delta' : 2}

if input_dim == 1:
    kwargs.update({'norm_multiplier' : 12.5})
    process_data_class = ProcessDataTbsPdfMF(**kwargs)
    cnn_kwargs = None
elif input_dim == 3:
    kwargs.update({'norm_multiplier' : 100})
    process_data_class = ProcessData3DPdfMF(**kwargs)
    # cnn_kwargs = {'in_features' : 1, 'num_cells' : [8, 4, 1], 'kernel_sizes' : 3}
    cnn_kwargs = {'in_features' : 1, 'num_cells' : [8, 1], 'kernel_sizes' : 3}

timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if timestamp=='' else timestamp


alg_name = ha_assig_alg_str+f'_{input_dim}d{tag}_training'

#---------------------
# The class ORANHAEnv is an interface to communicate with the O-RAN experimental platform usign the Gym interface
# env = ORANHAEnv(ha_config_file, bs_config_file_training, ha_assig_alg_str, process_data_class, 
#                    restart, max_running_time, action_range, action_dim, timestamp=timestamp, 
#                    policy_name=alg_name)
env = None
#---------------------

actor_kwargs = {'num_cells' : [256, 256], 'activation_class' : nn.ReLU, 'last_activation' : nn.Sigmoid}
critic_kwargs = {'num_cells' : [256, 256], 'activation_class' : nn.ReLU}

if noisyNet:
    # actor_kwargs.update({'layer_class' : NoisyLinear})
    noise = None
else:
    noise = OUNoise(env.n_bss, decay_period=train_iters*training_epochs)


if ml_version == 'mf_v1':
    deep_cb = ECORAN_MF(env, noise, batch_size, actor_kwargs=actor_kwargs, 
                     critic_kwargs=critic_kwargs, convnet_kwargs=cnn_kwargs,
                     updates_per_step=updates_per_step, device=device)
# elif ml_version == 'IL':
#     deep_cb = ECORAN_IL(env, noise, batch_size, actor_kwargs=actor_kwargs, 
#                      critic_kwargs=critic_kwargs, convnet_kwargs=cnn_kwargs,
#                      updates_per_step=updates_per_step, device=device)
# elif ml_version == 'maddpg':
#     deep_cb = ECORAN_MADDPG(env, noise, batch_size, actor_kwargs=actor_kwargs, 
#                      critic_kwargs=critic_kwargs, convnet_kwargs=cnn_kwargs,
#                      updates_per_step=updates_per_step, device=device)
else:
    raise Exception("ML algorithm not found.")



for _ in range(training_epochs):
    deep_cb.run(train_iters)
    deep_cb.env.reset() # reset the experimental platform env for the next training epoch


######## Evaluation ########

alg_name = ha_assig_alg_str+f'_{input_dim}d{tag}'

#---------------------
# eval_env = ORANHAEnv(ha_config_file, bs_config_file, ha_assig_alg_str, process_data_class, 
#                     restart, max_running_time, action_range, action_dim, timestamp=timestamp, policy_name=alg_name)
eval_env = None
#---------------------

deep_cb.env = eval_env
deep_cb.run(eval_iters, evaluation=1)



######## Save config data ########
config_file = os.path.join(os.getcwd(), 'results', timestamp, timestamp+f'_{alg_name}_config.json')

config_data = {'name' : alg_name, 'policy' : policy, 'ha_config_file' : ha_config_file, 'bs_config_file' : bs_config_file, 
               'bs_config_file_training' : bs_config_file_training, 'input_dim': input_dim, 'updates_per_step' : updates_per_step, 
               'training_epochs' : training_epochs, 'noisyNet' : noisyNet, 'device' : str(device), 
               'process_data_class' : str(type(process_data_class)), 'process_data_class_kwargs' : str(kwargs), 
               'cnn_kwargs' : str(cnn_kwargs), 'actor_kwargs' : str(actor_kwargs), 'critic_kwargs' : str(critic_kwargs), 
               'exploration_noise' : str(type(noise)), 'norm_contex_max_val' : process_data_class.max_val,
               'n_context_av' : n_context_av, 'ml_version' : ml_version}

with open(config_file, 'w') as f:
    json.dump(config_data, f, indent=4)
    
######## Benchmark Evaluation ########

if bench_eval:
    benchmarks = ['onlyCPU', 'onlyGPU', 'MWT']
        
    for bench in benchmarks:
        print(f'Evaluation {bench} policy...')
        # eval_env = ORANHAEnv(ha_config_file, bs_config_file, bench, process_data_class, 
        #                    restart, max_running_time, action_range=(0, 0), timestamp=timestamp, policy_name=bench)
        eval_env = None
        for step in range(eval_iters):
            _, _, done, _ = eval_env.step(None) 
            if done:
                break
        eval_env.close()
