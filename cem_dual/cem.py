import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3 as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no
from collections import OrderedDict
from utils import save_data, load_data
import os
import os.path as osp
import matplotlib.pyplot as plt

# choose the env
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--exp_type', default="sim", help='choose sim or real')
args = parser.parse_args()

# Simulation Environment
env = gym_custom.make('dual-ur3-larr-for-train-v0')
servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
gripper_scale_factor = np.array([1.0])
env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)

# Real Environment
if args.exp_type == 'real':
    real_env = gym_custom.make('dual-ur3-larr-real-for-train-v0',
        host_ip_right='192.168.5.102',
        host_ip_left='192.168.5.101',
        rate=25
    )
    servoj_args, speedj_args = {'t': 2/real_env.rate._freq, 'wait': False}, {'a': 1, 't': 4/real_env.rate._freq, 'wait': False}
    # 1. Set initial as current configuration
    real_env.set_initial_joint_pos('current')
    real_env.set_initial_gripper_pos('current')
    # 2. Set inital as default configuration
    real_env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
    real_env.set_initial_gripper_pos(np.array([0.0, 0.0]))

    time.sleep(1.0)


# (참고용) Action limits 
COMMAND_LIMITS = {
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])*0.1,
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])*0.1], # [rad/s]
    'move_gripper': [np.array([-1]), np.array([1])] # [0: open, 1: close]
}

# Pre-defined action sequence
action_seq = np.array([[0,0,0,0,0,0,0.3,0,0,0,0,0]]*100+[[0,0,0,0,0,0,-0.3,0,0,0,0,0]]*100+\
                      [[0,0,0,0,0,0,0,0.3,0,0,0,0]]*100+[[0,0,0,0,0,0,0,-0.3,0,0,0,0]]*100+\
                      [[0,0,0,0,0,0,0,0,0.3,0,0,0]]*100+[[0,0,0,0,0,0,0,0,-0.3,0,0,0]]*100+\
                      [[0,0,0,0,0,0,0,0,0,0.6,0,0]]*50 +[[0,0,0,0,0,0,0,0,0,-0.6,0,0]]*50+\
                      [[0,0,0,0,0,0,0,0,0,0,0.6,0]]*100+[[0,0,0,0,0,0,0,0,0,0,-0.6,0]]*100+\
                      [[0,0,0,0,0,0,0,0,0,0,0,0.6]]*100+[[0,0,0,0,0,0,0,0,0,0,0,-0.6]]*100)

# Run simulation
# if real, get the data
if args.exp_type == 'real':
    real_data = []
    state = real_env.reset()
    # env.wrapper_right.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]
    # env.wrapper_left.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]
    for i in range(1100):
        next_state, reward, done, _  = real_env.step({
            'right': {
                'speedj': {'qd': action_seq[i][:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': action_seq[i][6:12], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })

        curr_pos = real_env.get_obs_dict()['left']['curr_pos']      # from real env
        real_data.append(curr_pos)
        # env.render()
    # Save real data
    real_data = np.array(real_data)
    save_data(real_data, "real_data_left.npy")


# if sim, RUN CEM
else:
    n_seq = 2
    n_horrizon =1100
    n_dim = 3
    n_iter = 1000
    n_elit = 1
    alpha = 0.9

    # a, P, I params # res if [5, 0.2, 10]
    lim_high = np.array([50, 50, 50, 50, 50, 50])
    lim_low  = np.array([0, 0, 0, 0, 0, 0])

    # load data
    sim_data = np.zeros([n_seq, n_horrizon, n_dim])
    real_data = load_data("cem_dual/data/real_data_left.npy")

    # logging
    logging = []
    logging_err = []

    # CEM
    for k in range(n_iter):

        # sample params
        if k == 0:
            candidate_parameters = lim_low + (lim_high - lim_low)*np.random.rand(n_seq, len(lim_high))
            prams_mean = np.mean(candidate_parameters, axis=0)
            prams_std = np.std(candidate_parameters, axis=0)
        else:
            candidate_parameters = np.random.normal(prams_mean, prams_std, (n_seq, len(lim_high)))

        # evaluate params
        for i in range(n_seq):
            state = env.reset()
            # ur3_scale_factor
            env.wrapper_left.ur3_scale_factor[:6]= candidate_parameters[i][:6]

            for j in range(n_horrizon):
                curr_pos = env.get_obs_dict()['left']['curr_pos']       # from sim env
                sim_data[i][j][:] = curr_pos
                next_state, reward, done, _  = env.step({
                    'right': {
                        'speedj': {'qd': action_seq[j][:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                        'move_gripper_force': {'gf': np.array([10.0])}
                    },
                    'left': {
                        'speedj': {'qd': action_seq[j][6:12], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                        'move_gripper_force': {'gf': np.array([10.0])}
                    }
                })

                # env.render()
        
        mse_results = np.zeros(n_seq)
        for i in range(n_seq):
            mse = np.mean((sim_data[i] - real_data) ** 2)
            mse_results[i] = mse

        # smallest MSE
        smallest_indices = np.argpartition(mse_results, n_elit)[:n_elit]

        # pick elites
        elite_params = candidate_parameters[smallest_indices]
        elite_err = sum(mse_results[smallest_indices])/n_elit

        # Update the elite mean and variance
        prams_mean = alpha * np.mean(elite_params, axis=0) + (1 - alpha) * prams_mean
        prams_std = alpha * np.std(elite_params, axis=0) + (1 - alpha) * prams_std
        logging.append(prams_mean)
        logging_err.append(elite_err)
        print(prams_mean)
        
        # Plot
        plt.figure(figsize=(8, 10))
        plt.clf()  
        history_array = np.array(logging).T  
        history_array_err = np.array(logging_err).T  

        ax1 = plt.subplot(5, 1, 1)   
        plt.plot(history_array[0], label='p1', marker='o')
        plt.plot(history_array[1], label='p2', marker='o')
        plt.plot(history_array[2], label='p3', marker='o')
        plt.plot(history_array[3], label='p4', marker='o')
        plt.plot(history_array[4], label='p5', marker='o')
        plt.plot(history_array[5], label='p6', marker='o')

        # plt.axhline(y=60, color='k', linestyle='--', label='p1')
        # plt.axhline(y=50, color='k', linestyle='--', label='p2')
        # plt.axhline(y=40, color='k', linestyle='--', label='p3')
        # plt.axhline(y=30, color='k', linestyle='--', label='p4')
        # plt.axhline(y=20, color='k', linestyle='--', label='p5')
        # plt.axhline(y=10, color='k', linestyle='--', label='p6')

        plt.title("History of Parameter/Error")
        plt.ylabel("parameter")
        plt.legend()

        ax2 = plt.subplot(5, 1, 2, sharex=ax1)  
        plt.plot(history_array_err, color='k', label='err', marker=".")
        plt.xlabel("Iteration")
        plt.ylabel("error")
        plt.legend()
    
        # for traj visualization, real vs sim
        logging_traj = []
        state = env.reset()
        env.wrapper_left.ur3_scale_factor[:6] = prams_mean[:6]
        # env.wrapper_left.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]
        for j in range(n_horrizon):
            curr_pos = env.get_obs_dict()['left']['curr_pos']       # from sim env

            next_state, reward, done, _  = env.step({
                'right': {
                    'speedj': {'qd': action_seq[j][:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                },
                'left': {
                    'speedj': {'qd': action_seq[j][6:12], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                }
            })

            logging_traj.append(curr_pos)

        # Plot
        history_array_traj = np.array(logging_traj).T 
        real_array_traj = np.array(real_data).T

        ax3 = plt.subplot(5, 1, 3)  
        plt.plot(history_array_traj[0], label='sim', marker=',')
        plt.plot(real_array_traj[0], label='real', linestyle='--')
        plt.xlabel("timestep")
        plt.ylabel("x$_{position}$")
        plt.legend()

        ax4 = plt.subplot(5, 1, 4)  
        plt.plot(history_array_traj[1], label='sim', marker=',')
        plt.plot(real_array_traj[1], label='real', linestyle='--')
        plt.xlabel("timestep")
        plt.ylabel("y$_{position}$")
        plt.legend()

        ax5 = plt.subplot(5, 1, 5)  
        plt.plot(history_array_traj[2], label='sim', marker=',')
        plt.plot(real_array_traj[2], label='real', linestyle='--')
        plt.xlabel("timestep")
        plt.ylabel("z$_{position}$")
        plt.legend()

        plt.pause(1)
        plt.close()

    plt.show() 
    





