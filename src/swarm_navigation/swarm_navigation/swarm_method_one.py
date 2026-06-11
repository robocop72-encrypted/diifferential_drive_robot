import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import re
from functools import partial

class SwarmAgentSimulation(Node):
    def __init__(self):
        super().__init__('swarm_agent_simulation')
        
        self.declare_parameter('robot_name', 'robot_1')
        param_name = self.get_parameter('robot_name').get_parameter_value().string_value
        
        ns = self.get_namespace().strip('/')
        node_name = self.get_name()
        
        if 'robot' in ns:
            base_style = ns
        elif 'robot' in node_name:
            base_style = node_name
        else:
            base_style = param_name

        numbers = [int(s) for s in re.findall(r'\d+', base_style)]
        robot_id = numbers[0] if numbers else 1

        if '_' in base_style:
            self.robot_name = f'robot_{robot_id}'
            self.all_robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4']
        else:
            self.robot_name = f'robot{robot_id}'
            self.all_robot_names = ['robot1', 'robot2', 'robot3', 'robot4']
            
        self.goal_x = 25.0
        self.goal_y = 25.0
        
        self.current_pose = [0.0, 0.0, 0.0]
        self.lidar_ranges = []
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.robot_radius = 0.20
        
        self.peer_positions = {}
        
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        
        try:
            self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.lidar_callback, qos_profile_sensor_data)
        except Exception:
            self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.lidar_callback, 10)
        
        for peer in self.all_robot_names:
            if peer != self.robot_name:
                self.peer_positions[peer] = None
                self.create_subscription(
                    Odometry, 
                    f'/{peer}/odom', 
                    partial(self.peer_odom_callback, peer_name=peer), 
                    10
                )
                
        self.loop_timer = self.create_timer(0.05, self.control_loop_step)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = [pos.x, pos.y, yaw]

    def peer_odom_callback(self, msg, peer_name):
        pos = msg.pose.pose.position
        self.peer_positions[peer_name] = [pos.x, pos.y]

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def control_loop_step(self):
        cx, cy, cyaw = self.current_pose
        
        dx = self.goal_x - cx
        dy = self.goal_y - cy
        dist_to_goal = np.hypot(dx, dy)
        
        if dist_to_goal < 0.8:
            self.stop_robot()
            return

        f_total_x = 0.0
        f_total_y = 0.0
        
        k_att = 1.5
        f_total_x += k_att * (dx / max(dist_to_goal, 0.1))
        f_total_y += k_att * (dy / max(dist_to_goal, 0.1))
        
        k_rep_wall = 5.0
        d_influence_wall = 1.0
        emergency_braking = False
        min_clearance_found = 10.0
        
        if len(self.lidar_ranges) > 0:
            for i, r in enumerate(self.lidar_ranges):
                if math.isnan(r) or math.isinf(r) or r <= 0.05:
                    continue
                
                clearance = r - self.robot_radius
                if clearance < min_clearance_found:
                    min_clearance_found = clearance
                
                ray_angle_rel = self.angle_min + (i * self.angle_increment)
                
                if abs(ray_angle_rel) < 0.60 and clearance < 0.15:
                    emergency_braking = True
                
                if clearance < d_influence_wall:
                    ray_angle_global = ray_angle_rel + cyaw
                    obs_x = cx + r * math.cos(ray_angle_global)
                    obs_y = cy + r * math.sin(ray_angle_global)
                    
                    clearance_clamped = max(clearance, 0.01)
                    rep_mag = k_rep_wall * (1.0 / clearance_clamped - 1.0 / d_influence_wall) / (clearance_clamped ** 2)
                    
                    vec_x = cx - obs_x
                    vec_y = cy - obs_y
                    v_len = np.hypot(vec_x, vec_y)
                    
                    if v_len > 0:
                        f_total_x += rep_mag * (vec_x / v_len)
                        f_total_y += rep_mag * (vec_y / v_len)

            k_rep_peer = 8.0
            d_influence_peer = 1.5
            double_radius = self.robot_radius * 2.0
            
            for peer, p_pos in self.peer_positions.items():
                if p_pos is not None:
                    px, py = p_pos
                    p_dist = np.hypot(cx - px, cy - py)
                    peer_clearance = p_dist - double_radius
                    
                    if peer_clearance < d_influence_peer and p_dist > 0.0:
                        if p_dist < double_radius + 0.15:
                            emergency_braking = True
                        
                        peer_clearance_clamped = max(peer_clearance, 0.01)
                        rep_mag = k_rep_peer * (1.0 / peer_clearance_clamped - 1.0 / d_influence_peer) / (peer_clearance_clamped ** 2)
                        
                        vec_x = cx - px
                        vec_y = cy - py
                        f_total_x += rep_mag * (vec_x / p_dist)
                        f_total_y += rep_mag * (vec_y / p_dist)

        desired_heading = math.atan2(f_total_y, f_total_x)
        heading_error = desired_heading - cyaw
        
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        msg = Twist()
        
        if emergency_braking:
            msg.linear.x = 0.0
            msg.angular.z = 0.4 if heading_error > 0 else -0.4
        elif abs(heading_error) > 1.0:
            msg.linear.x = 0.0
            msg.angular.z = 0.4 if heading_error > 0 else -0.4
        else:
            base_speed = 0.3
            speed_factor = min_clearance_found / d_influence_wall
            speed_factor = max(0.1, min(1.0, speed_factor))
            
            msg.linear.x = base_speed * speed_factor * math.cos(heading_error)
            msg.angular.z = 1.5 * heading_error

        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SwarmAgentSimulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()