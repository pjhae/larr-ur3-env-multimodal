
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


## 1. Register your own rigid body in optitrack(ex, cube_rainbow, cube_RB)
## 2. Get the position of rigid body using rospy.wait_for_message


def listener_wait_msg():

    rospy.init_node('ros_subscription_test_node')
    cube1_msg = rospy.wait_for_message('optitrack/cube_rainbow/poseStamped', PoseStamped)
    cube2_msg = rospy.wait_for_message('optitrack/cube_RB/poseStamped', PoseStamped)
    return cube1_msg.pose.position, cube2_msg.pose.position

if __name__ == '__main__':

    msg_count = 0

    while True:
 
        cube1_pos, cube2_pos = listener_wait_msg()
        cube1_pos_array = np.array([cube1_pos.x, cube1_pos.y, cube1_pos.z]) - np.array([ 0.83438993 ,-0.14621475 , 0.76058847] ) + np.array([0.45, -0.325, 0.8])
        cube2_pos_array = np.array([cube2_pos.x, cube2_pos.y, cube2_pos.z]) - np.array([-1.05037975 ,-0.41750526,  0.76240301] ) + np.array([0.45, -0.325, 0.8])
        print("ctrl : ", cube1_pos_array, cube2_pos_array)
 

        if msg_count >= 10000: break

    print('Done!')
