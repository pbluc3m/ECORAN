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
