import rospy
from geometry_msgs.msg import Twist

def move_robot():
    # init ROS node
    rospy.init_node('robot_controller', anonymous=True)
    
    # make publish /cmd_vel topic
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    move_cmd = Twist()
    
    # Set linear velocity and angular velocity.
    move_cmd.linear.x = 0.5  # Linear velocity 0.5 m/s, indicating forward movement.
    move_cmd.angular.z = 0.0  # Angular velocity 0.2 rad/s, indicating rotation.

    # set publish freq
    rate = rospy.Rate(10)  #  10Hz

    rospy.loginfo("Robot moving...")
    for _ in range(50): 
        pub.publish(move_cmd)
        rate.sleep()  # Control publish frequency is set to 10 Hz.

    # stop
    rospy.loginfo("Robot stopped.")
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.0
    pub.publish(move_cmd) 

if __name__ == "__main__":
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass