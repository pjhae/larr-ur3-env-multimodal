
# General package
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import time
import sys
import os
import os.path as osp

# Environment related
import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3 as URScriptWrapper
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no
from collections import OrderedDict
from get_data.utils import generate_action_sequence_2d, generate_action_sequence_3d
from get_data.utils import UprightConstraint
from get_data.utils import listener_wait_msg

# ROS related
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped



################## USER OPTION ################## 

# Rendering 
render = True # if exp_type is real, render should be FALSE

# Posture constraint
null_obj_func = UprightConstraint()

# Max velocity
max_velocity = (0.04, 0.04, 0)

# Type of experiment
exp_type = "real" # "sim" is not implemented yet

#################################################


# Environment
if exp_type == "sim":
    env = gym_custom.make('single-ur3-xy-larr-for-train-v0')
    servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}

elif exp_type == "real":
    env = gym_custom.make('single-ur3-larr-real-for-train-v0', # TODO
        host_ip_right='192.168.5.102',
        rate=20
    )
    servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 1, 't': 4/env.rate._freq, 'wait': False}
    # 1. Set initial as current configuration
    env.set_initial_joint_pos('current')
    env.set_initial_gripper_pos('current')
    # 2. Set inital as default configuration
    # env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0]))
    env.set_initial_joint_pos(np.array([ 1.22096933, -1.3951761, 1.4868261, -2.01667739, 0.84679318, -0.00242263]))
    env.set_initial_gripper_pos(np.array([255.0]))
    assert render is False

else:
    print("Please choose sim or real")

obs = env.reset()
dt = env.dt

if exp_type == "sim":
    PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    # scale factor
    env.wrapper_right.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]

elif exp_type == "real":
        env.env = env

if exp_type == "real":
    if prompt_yes_or_no('current qpos is \r\n right: %s deg?\r\n'
        %(np.rad2deg(env.env._init_qpos[:6]))) is False:
        print('exiting program!')
        env.close()
        sys.exit()
time.sleep(1.0)


while True:
    state = env.reset()
    state[:3] = np.array([0.45, -0.325, 0.8])
    state = state[:3]
    
    episode_reward = 0
    step = 0
    done = False

    while not done:

        # # # # ROS related
        # goal_pos = listener_wait_msg("goal_cube")
        # red_pos = listener_wait_msg("red_cube")
        # blue_pos = listener_wait_msg("blue_cube")
        # calib_offset_goal =
        # calib_offset_block =  
        # goal_pos -= calib_offset_goal
        # block_pos -= calib_offset_block

        goal_pos = np.array([0.45, -0.325, 0.8])
        curr_pos = np.concatenate([state[:2],[0.8]])
        action_squence, _ = generate_action_sequence_3d(curr_pos, goal_pos, max_velocity)
        q_right_des, _ ,_ ,_ = env.inverse_kinematics_ee(curr_pos+ action_squence[0], null_obj_func, arm='right')
        dt = 1
        qvel_right = (q_right_des - env.get_obs_dict()['right']['qpos'])/dt

        next_state, _, _, _  = env.step({
            'right': {
                'speedj': {'qd': qvel_right, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([15.0])}
            }
        })
 
        if render == True :
            env.render()

        step += 1
        state = next_state[:3]

        if exp_type == "real" and step == 1000:
            break   
    





