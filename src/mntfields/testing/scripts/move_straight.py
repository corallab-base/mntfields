#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MoveStraight(Node):
    def __init__(self):
        super().__init__('move_straight')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Move straight 1m node started!')
        
    def move_distance(self, distance=1.0, speed=0.2):
        """Move forward distance meters at given speed"""
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = 0.0
        
        start_time = time.time()
        duration = distance / speed  # Time needed
        
        self.get_logger().info(f'Moving {distance}m forward at {speed}m/s for {duration:.1f}s')
        
        while rclpy.ok() and (time.time() - start_time) < duration:
            self.publisher.publish(cmd)
            time.sleep(0.1)
        
        # Stop
        cmd.linear.x = 0.0
        self.publisher.publish(cmd)
        self.get_logger().info('Stopped!')

def main(args=None):
    rclpy.init(args=args)
    mover = MoveStraight()
    
    try:
        mover.move_distance(1.0, 0.2)  # 1 meter at 0.2 m/s
    except KeyboardInterrupt:
        pass
    
    mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
