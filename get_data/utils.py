
import numpy as np
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


def generate_action_sequence_2d(start_point, end_point, max_distance):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    num_steps_x = int(abs(dx) / max_distance[0])  # x axis
    num_steps_y = int(abs(dy) / max_distance[1])  # y axis
    num_steps = max(num_steps_x, num_steps_y)  # choose larger one

    action_sequence = []

    if num_steps > 0:
        step_x = dx / (num_steps+1)
        step_y = dy / (num_steps+1)

        for _ in range(num_steps):
            action_sequence.append((step_x, step_y))

    if num_steps == 0 :
        action_sequence.append((dx, dy))
        return action_sequence, 1
    
    # Calculate the remaining distance in each dimension
    remaining_distance_x = end_point[0] - (start_point[0] + step_x * num_steps)
    remaining_distance_y = end_point[1] - (start_point[1] + step_y * num_steps)

    action_sequence.append((remaining_distance_x, remaining_distance_y))

    return np.array(action_sequence), len(action_sequence)

def generate_action_sequence_3d(start_point, end_point, max_distance):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    dz = end_point[2] - start_point[2]

    num_steps_x = int(abs(dx) / max_distance[0])  # x axis
    num_steps_y = int(abs(dy) / max_distance[1])  # y axis
    num_steps_z = int(abs(dz) / max_distance[2])  # z axis
    num_steps = max(num_steps_x, num_steps_y, num_steps_z)  # choose the largest one

    action_sequence = []

    if num_steps > 0:
        step_x = dx / (num_steps + 1)
        step_y = dy / (num_steps + 1)
        step_z = dz / (num_steps + 1)

        for _ in range(num_steps):
            action_sequence.append((step_x, step_y, step_z))

    if num_steps == 0:
        action_sequence.append((dx, dy, dz))
        return action_sequence, 1

    # Calculate the remaining distance in each dimension
    remaining_distance_x = end_point[0] - (start_point[0] + step_x * num_steps)
    remaining_distance_y = end_point[1] - (start_point[1] + step_y * num_steps)
    remaining_distance_z = end_point[2] - (start_point[2] + step_z * num_steps)

    action_sequence.append((remaining_distance_x, remaining_distance_y, remaining_distance_z))

    return np.array(action_sequence), len(action_sequence)



# Constraint
class UprightConstraint(NullObjectiveBase):
    
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)
    


# Ros
def listener_wait_msg(cube_name):
    topic_name = f'optitrack/{cube_name}/poseStamped'
    rospy.init_node('ros_subscription_test_node')
    cube_msg = rospy.wait_for_message(topic_name, PoseStamped)
    return cube_msg.pose.position