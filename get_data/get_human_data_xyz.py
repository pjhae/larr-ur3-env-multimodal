
# General package
import argparse
from pynput import keyboard

import datetime
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
# from get_data.utils import listener_wait_msg
from get_data.ros_test import listener_wait_msg


# ROS related
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


#################################### USER OPTION #################################### 

# Rendering 
render = False # if exp_type is real, render should be FALSE

# Posture constraint
null_obj_func = UprightConstraint()

# Max velocity
max_velocity = (0.06, 0.06, 0.06)

# Type of experiment
exp_type = "real" # "sim" is not implemented yet

##################################################################################### 


# Keyboard
c_key_pressed = False # 각 키에 대한 상태를 추적하기 위한 변수
o_key_pressed = False

def on_press(key): # 키가 눌렸을 때 실행할 함수
    global c_key_pressed, o_key_pressed
    if key == keyboard.KeyCode.from_char('c'):
        c_key_pressed = True
    elif key == keyboard.KeyCode.from_char('o'):
        o_key_pressed = True

def on_release(key): # 키가 놓였을 때 실행할 함수
    global c_key_pressed, o_key_pressed
    if key == keyboard.KeyCode.from_char('c'):
        c_key_pressed = False
    elif key == keyboard.KeyCode.from_char('o'):
        o_key_pressed = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release) # 키보드 리스너 시작
listener.start()


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



# Main loop

while True:
    state = env.reset()
    state[:3] = np.array([0.45, -0.325, 0.8])
    state = state[:3]
    
    episode_reward = 0
    step = 0
    done = False


    while not done:

            ####  ROS related
            # goal_pos = listener_wait_msg("goal_cube")
            # red_pos = listener_wait_msg("red_cube")
            # blue_pos = listener_wait_msg("blue_cube")
            # calib_offset_goal = TODO
            # calib_offset_block = TODO
            # goal_pos -= calib_offset_goal
            # block_pos -= calib_offset_block
            
            
            cube1_pos, cube2_pos = listener_wait_msg()
            cube2_pos_array = np.array([cube2_pos.x, cube2_pos.y, cube2_pos.z]) - np.array([-1.05037975 ,-0.41750526,  0.76240301] ) + np.array([0.45, -0.325, 0.8])
            
            goal_pos = cube2_pos_array 
            
            # if step < 200:
            #     goal_pos = np.array([0.0, -0.325, 0.8])
            # if step > 200 and step < 400:
            #     goal_pos = np.array([0.0, -0.325, 1.0])
            # if step > 400:
            #     goal_pos = np.array([0.45, -0.325, 0.8])

            curr_pos = np.concatenate([state[:3]])
            action_squence, _ = generate_action_sequence_3d(curr_pos, goal_pos, max_velocity)
            action = action_squence[0]
            q_right_des, _ ,_ ,_ = env.inverse_kinematics_ee(curr_pos + action, null_obj_func, arm='right')
            dt = 1
            qvel_right = (q_right_des - env.get_obs_dict()['right']['qpos'])/dt

            next_state, _, _, _  = env.step({
                'right': {
                    'speedj': {'qd': qvel_right, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([15.0])}
                }
            })

            step += 1
            state = next_state[:3]

            if c_key_pressed:
                # 'c' 키를 눌러 그리퍼를 닫기
                print("close_gripper")
                env.step({'right': {'close_gripper': {}}})
                time.sleep(3.0)

            if o_key_pressed:
                # 'o' 키를 눌러 그리퍼를 열기
                print("open_gripper")
                env.step({'right': {'open_gripper': {}}})
                time.sleep(3.0)
                
            if render == True :
                env.render()


            if step == 10000:
                break




