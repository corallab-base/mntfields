import select
import sys
import termios
import tty

import rclpy
from std_msgs.msg import Bool


def main():
    rclpy.init()
    node = rclpy.create_node('tb4_nav_keyboard')
    pub = node.create_publisher(Bool, '/mntfields/nav_enable', 10)

    if not sys.stdin.isatty():
        node.get_logger().error('stdin is not a TTY; run this in a terminal.')
        node.destroy_node()
        rclpy.shutdown()
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    enabled = False
    node.get_logger().info('TB4 nav keyboard ready: Enter=start/resume, c=stop, q=quit')
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            ch = sys.stdin.read(1)
            if ch in ('\n', '\r'):
                pub.publish(Bool(data=True))
                if not enabled:
                    node.get_logger().info('Published start/resume')
                    enabled = True
            elif ch in ('c', 'C'):
                pub.publish(Bool(data=False))
                if enabled:
                    node.get_logger().info('Published stop')
                    enabled = False
            elif ch in ('q', 'Q'):
                break
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
