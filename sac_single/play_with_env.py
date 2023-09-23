import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3 as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from collections import OrderedDict
import numpy as np
import itertools


######################## CHECKLIST FOR SEUNGJAE ########################
##
## 1. action space 가 delx, dely로 2차원이지만, 93번째 줄에 대입할 때에는 IK 풀어야 해서 3차원으로 줘야함 (delz 값 꼭 0으로 하기)
##    그래서 저는 38번째 줄처럼 3차원으로 구현했습니다. 드린 데이터의 액션은 2차원으로 드렸습니다.
##
## 2. 본 환경인 SingleUR3XYCOMBEnv환경에서 get_obs함수는 다양한 값들을 return 하므로, 우리가 쓸 state중에서 앞에서 6개 state[:6] 만 쓰면 됩니다. (92, 118번째 줄 참고)
##    순서대로 로봇의 현 x,y + 빨강블럭의 x,y + 파란블럭2의 x,y (7번째 부터는 Joint 관련 값들)
##
## 3. 시작 때부터 IK 정확하게 풀게 하기 위해서 state[:6] = np.array([0.45, -0.325, 0.3, -0.25, 0.3, -0.40]) 해줬습니다. (실제 시작위치 고려)
##
## 4. 90번째 줄 다양하게 해보면서 감 잡기!
##
########################################################################


# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym_custom.make('single-ur3-xy-left-comb-larr-for-train-v0')
servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
ur3_scale_factor = np.array([5,5,5,5,5,5])
gripper_scale_factor = np.array([1.0])
env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)

# Max episode
max_episode_steps = 600

# For action bound
COMMAND_LIMITS = {
    'movej': [np.array([-0.04, -0.04, 0]),
        np.array([0.04, 0.04, 0])] # [m]
}

def convert_action_to_space(action_limits):
    if isinstance(action_limits, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_action_to_space(value))
            for key, value in COMMAND_LIMITS.items()
        ]))
    elif isinstance(action_limits, list):
        low = action_limits[0]
        high = action_limits[1]
        space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
    else:
        raise NotImplementedError(type(action_limits), action_limits)
    return space

def _set_action_space():
    return convert_action_to_space({'_': COMMAND_LIMITS})

action_space = _set_action_space()['movej']


# Set motor gain scale
env.wrapper_right.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]

# End effector Constraint
class UprightConstraint(NullObjectiveBase):
    
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)
    
null_obj_func = UprightConstraint()


# Training Loop
total_numsteps = 0
updates = 0

# Train
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    state[:6] = np.array([0.45, -0.325, 0.3, -0.25, 0.3, -0.40])
    state = state[:6]
    
    while not done:

        action = action_space.sample()  # Sample random action
        # action = [-0.03, 0.0, 0.0]    # 이렇게 주면 delx가 음수니까 랜더링 해보면 계속 왼쪽 방향으로 감!
        env.render()
        curr_pos = np.concatenate([state[:2],[0.8]])
        q_right_des, _ ,_ ,_ = env.inverse_kinematics_ee(curr_pos + action, null_obj_func, arm='right')
        dt = 1
        qvel_right = (q_right_des - env.get_obs_dict()['right']['qpos'])/dt

        next_state, reward, done, _  = env.step({
            'right': {
                'speedj': {'qd': qvel_right, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([15.0])}
            }
        })

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon. (max timestep 되었다고 done 해서 next Q = 0 되는 것 방지)
        mask = 1 if episode_steps == max_episode_steps else float(not done)

        state = next_state[:6]


    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

env.close()


