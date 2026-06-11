import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
import cv2
import json
import time
from cv_bridge import CvBridge

class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.network(x)

class MultiAgentRLManager(Node):
    def __init__(self):
        super().__init__('multi_agent_rl_manager')
        
        self.bridge = CvBridge()
        self.robot_names = ['robot_1', 'robot_2', 'robot_3', 'robot_4']
        self.k_neighbors = 2
        
        self.goal_x = 3.0
        self.goal_y = 3.0
        
        self.state_dim = 11 + (self.k_neighbors * 2)
        self.action_dim = 3
        
        self.shared_brain = DeepQNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.shared_brain.parameters(), lr=0.0003)
        self.memory = list()
        self.memory_max_size = 60000
        self.criterion = nn.MSELoss()
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.02
        self.gamma = 0.99
        self.batch_size = 128
        
        self.poses = {name: [0.0, 0.0, 0.0] for name in self.robot_names}
        self.lidars = {name: np.ones(5, dtype=np.float32) for name in self.robot_names}
        self.states = {name: np.zeros(self.state_dim, dtype=np.float32) for name in self.robot_names}
        self.actions = {name: 0 for name in self.robot_names}
        self.last_rewards = {name: 0.0 for name in self.robot_names}
        self.stuck_counters = {name: 0 for name in self.robot_names}
        self.cam_frame_counters = {name: 0 for name in self.robot_names}
        
        self.episode_paths = {name: [] for name in self.robot_names}
        self.best_path_length = float('inf')
        self.best_path_file = "best_path_log.json"
        
        self.user_pose = [0.0, 0.0, 0.0]
        self.stuck_threshold = 35
        self.episode_counter = 1
        self.weight_file = "swarm_brain.pth"
        self.is_resetting = False
        
        self.colors = {
            'robot_1': (255, 0, 0),
            'robot_2': (0, 255, 255),
            'robot_3': (255, 0, 255),
            'robot_4': (0, 165, 255)
        }
        
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.user_odom_sub = self.create_subscription(Odometry, '/user_pose', self.user_pose_callback, 10)
        
        self.pubs = {}
        for name in self.robot_names:
            self.create_subscription(LaserScan, f'/{name}/scan', lambda msg, n=name: self.scan_callback(msg, n), 10)
            self.create_subscription(Odometry, f'/{name}/odom', lambda msg, n=name: self.odom_callback(msg, n), 10)
            self.create_subscription(Image, f'/{name}/camera/image_raw', lambda msg, n=name: self.cam_callback(msg, n), 10)
            self.pubs[name] = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
            
        self.load_saved_weights()
        self.load_best_path()
        self.create_timer(0.1, self.rl_step_loop)

    def load_saved_weights(self):
        if os.path.exists(self.weight_file):
            try:
                ckpt = torch.load(self.weight_file)
                self.shared_brain.load_state_dict(ckpt['model_state'])
                self.epsilon = ckpt['epsilon']
                self.episode_counter = ckpt['episode']
                self.get_logger().info(f'Resuming Swarm from Episode {self.episode_counter}')
            except Exception as e:
                self.get_logger().warn(f'Checkpoints bypassed: {str(e)}')

    def save_current_weights(self):
        try:
            ckpt = {'model_state': self.shared_brain.state_dict(), 'epsilon': self.epsilon, 'episode': self.episode_counter}
            torch.save(ckpt, self.weight_file)
        except Exception:
            pass

    def load_best_path(self):
        if os.path.exists(self.best_path_file):
            try:
                with open(self.best_path_file, 'r') as f:
                    data = json.load(f)
                    self.best_path_length = data.get('length', float('inf'))
                    self.get_logger().info(f'Loaded optimal legacy path benchmark. Best length: {self.best_path_length:.2f}m')
            except Exception:
                pass

    def cam_callback(self, msg, robot_name):
        if self.is_resetting:
            return
        self.cam_frame_counters[robot_name] += 1
        if self.cam_frame_counters[robot_name] % 3 != 0:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            optimized_image = cv2.resize(cv_image, (240, 180), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"{robot_name} View", optimized_image)
            cv2.waitKey(1)
        except Exception:
            pass

    def scan_callback(self, msg, robot_name):
        if self.is_resetting:
            return
        src_ranges = np.array(msg.ranges, dtype=np.float32)
        src_ranges[~np.isfinite(src_ranges)] = 12.0
        
        split_segments = np.array_split(src_ranges, 5)
        self.lidars[robot_name] = np.array([seg.min() for seg in split_segments], dtype=np.float32)
        
        if self.lidars[robot_name].min() < 0.28:
            self.stuck_counters[robot_name] += 1
        else:
            if self.stuck_counters[robot_name] > 0:
                self.stuck_counters[robot_name] -= 1

    def get_yaw_from_quat(self, quat):
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg, robot_name):
        if self.is_resetting:
            return
        pos = msg.pose.pose.position
        yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        self.poses[robot_name] = [pos.x, pos.y, yaw]

    def user_pose_callback(self, msg):
        if self.is_resetting:
            return
        pos = msg.pose.pose.position
        yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        self.user_pose = [pos.x, pos.y, yaw]

    def construct_state(self, name):
        current_pose = self.poses[name]
        lidar_data = self.lidars[name]
        last_action = self.actions[name]
        last_reward = self.last_rewards[name]
        
        dx_g = self.goal_x - current_pose[0]
        dy_g = self.goal_y - current_pose[1]
        dist_g = math.hypot(dx_g, dy_g)
        angle_g = math.atan2(dy_g, dx_g) - current_pose[2]
        angle_g = math.atan2(math.sin(angle_g), math.cos(angle_g))
        
        dx_u = self.user_pose[0] - current_pose[0]
        dy_u = self.user_pose[1] - current_pose[1]
        dist_u = math.hypot(dx_u, dy_u)
        angle_u = math.atan2(dy_u, dx_u) - current_pose[2]
        angle_u = math.atan2(math.sin(angle_u), math.cos(angle_u))
        
        peer_distances = []
        for peer_name in self.robot_names:
            if peer_name == name:
                continue
            dx_p = self.poses[peer_name][0] - current_pose[0]
            dy_p = self.poses[peer_name][1] - current_pose[1]
            d_p = math.hypot(dx_p, dy_p)
            a_p = math.atan2(dy_p, dx_p) - current_pose[2]
            a_p = math.atan2(math.sin(a_p), math.cos(a_p))
            peer_distances.append((d_p, a_p))
            
        peer_distances.sort(key=lambda x: x[0])
        
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0:5] = lidar_data
        state[5] = dist_g
        state[6] = angle_g
        state[7] = dist_u
        state[8] = angle_u
        state[9] = float(last_action)
        state[10] = float(last_reward)
        
        idx = 11
        for i in range(self.k_neighbors):
            if i < len(peer_distances):
                state[idx] = peer_distances[i][0]
                state[idx+1] = peer_distances[i][1]
            else:
                state[idx] = 15.0
                state[idx+1] = 0.0
            idx += 2
            
        return state

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.shared_brain(state_t)
        return torch.argmax(q_values).item()

    def execute_action(self, name, action_idx):
        twist = Twist()
        if action_idx == 0:
            twist.linear.x = 0.22
            twist.angular.z = 0.0
        elif action_idx == 1:
            twist.linear.x = 0.08
            twist.angular.z = 0.45
        elif action_idx == 2:
            twist.linear.x = 0.08
            twist.angular.z = -0.45
        self.pubs[name].publish(twist)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q = self.shared_brain(states_t).gather(1, actions_t)
        max_next_q = self.shared_brain(next_states_t).detach().max(1)[0].unsqueeze(1)
        target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.memory_max_size:
            self.memory.pop(0)

    def render_path_visualizer(self):
        if self.is_resetting:
            return
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 20
        
        gx = int((self.goal_x + 5.0) * 50)
        gy = int((5.0 - self.goal_y) * 50)
        cv2.circle(canvas, (gx, gy), 15, (0, 255, 0), -1)
        cv2.putText(canvas, "GOAL", (gx - 18, gy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        for name in self.robot_names:
            coords = self.episode_paths[name]
            if len(coords) < 2:
                continue
            color = self.colors.get(name, (255, 255, 255))
            for i in range(1, len(coords)):
                pt1 = (int((coords[i-1][0] + 5.0) * 50), int((5.0 - coords[i-1][1]) * 50))
                pt2 = (int((coords[i][0] + 5.0) * 50), int((5.0 - coords[i][1]) * 50))
                cv2.line(canvas, pt1, pt2, color, 2)
                
            rx = int((self.poses[name][0] + 5.0) * 50)
            ry = int((5.0 - self.poses[name][1]) * 50)
            cv2.circle(canvas, (rx, ry), 5, color, -1)
            
        cv2.putText(canvas, f"Episode: {self.episode_counter}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Best Path: {self.best_path_length:.2f}m", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Live Path Tracker Map", canvas)
        cv2.waitKey(1)

    def check_and_record_path_efficiency(self, winner_name):
        path = self.episode_paths[winner_name]
        if len(path) < 2:
            return
            
        total_dist = 0.0
        for i in range(1, len(path)):
            total_dist += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            
        if total_dist < self.best_path_length:
            self.best_path_length = total_dist
            path_data = {"episode": self.episode_counter, "winner": winner_name, "length": total_dist, "coordinates": path}
            try:
                with open(self.best_path_file, 'w') as f:
                    json.dump(path_data, f)
                self.get_logger().error(f'NEW MOST EFFICIENT PATH FOUND! Length: {total_dist:.2f} meters by {winner_name}')
            except Exception:
                pass

    def trigger_auto_reset(self, outcome_msg):
        self.is_resetting = True
        self.get_logger().error(f'Status: {outcome_msg} | Commencing Stabilized Reset Sequence...')
        
        stop_twist = Twist()
        for name in self.robot_names:
            self.pubs[name].publish(stop_twist)
            
        time.sleep(0.15)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.save_current_weights()
        
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Connecting to simulator reset server...')
            
        req = Empty.Request()
        self.reset_client.call(req)
        
        time.sleep(0.4)
        
        for name in self.robot_names:
            self.stuck_counters[name] = 0
            self.last_rewards[name] = 0.0
            self.episode_paths[name].clear()
            self.poses[name] = [0.0, 0.0, 0.0]
            self.lidars[name] = np.ones(5, dtype=np.float32)
            
        self.episode_counter += 1
        self.is_resetting = False

    def rl_step_loop(self):
        if self.is_resetting:
            return
            
        next_states = {}
        for name in self.robot_names:
            next_states[name] = self.construct_state(name)
            self.episode_paths[name].append((self.poses[name][0], self.poses[name][1]))
            
        any_won = False
        winner_name = None
        wall_impact_name = None
        swarm_collision = False
        
        for name in self.robot_names:
            if next_states[name][5] < 0.35:
                any_won = True
                winner_name = name
            if self.stuck_counters[name] >= self.stuck_threshold:
                wall_impact_name = name
                
        for i, name_a in enumerate(self.robot_names):
            for name_b in self.robot_names[i+1:]:
                dx = self.poses[name_a][0] - self.poses[name_b][0]
                dy = self.poses[name_a][1] - self.poses[name_b][1]
                if math.hypot(dx, dy) < 0.42:
                    swarm_collision = True
                    
        if any_won or (wall_impact_name is not None) or swarm_collision:
            if any_won:
                self.check_and_record_path_efficiency(winner_name)
                
            for name in self.robot_names:
                if any_won:
                    r = 100.0
                    msg = f"SWARM GOAL SUCCESS BY {winner_name.upper()}!"
                elif swarm_collision:
                    r = -25.0
                    msg = "SWARM INTERNAL COLLISION!"
                else:
                    r = -15.0 if name == wall_impact_name else -2.0
                    msg = f"WALL COLLISION BY {wall_impact_name.upper()}!"
                    
                self.add_to_memory((self.states[name], self.actions[name], r, next_states[name], 1.0))
                
            self.train_step()
            self.trigger_auto_reset(msg)
            return
            
        next_actions = {}
        for name in self.robot_names:
            next_actions[name] = self.select_action(next_states[name])
            self.execute_action(name, next_actions[name])
            
        for name in self.robot_names:
            prox_reward = (self.states[name][5] - next_states[name][5]) * 20.0
            dir_reward = math.cos(next_states[name][6]) * 0.4
            lidar_penalty = -0.15 / (self.lidars[name].min() + 0.02)
            
            reward = prox_reward + dir_reward + lidar_penalty
            
            for peer_name in self.robot_names:
                if peer_name == name:
                    continue
                dx = self.poses[peer_name][0] - self.poses[name][0]
                dy = self.poses[peer_name][1] - self.poses[name][1]
                p_dist = math.hypot(dx, dy)
                if p_dist < 0.8:
                    reward -= 0.4 / (p_dist + 0.1)
                    
            self.add_to_memory((self.states[name], self.actions[name], reward, next_states[name], 0.0))
            self.states[name] = next_states[name]
            self.actions[name] = next_actions[name]
            self.last_rewards[name] = reward
            
        self.train_step()
        self.render_path_visualizer()

def main(args=None):
    rclpy.init(args=args)
    node = MultiAgentRLManager()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()