import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import math

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class GazeboQLearningNode(Node):
    def __init__(self):
        super().__init__('gazebo_dqn_node')
        
        self.callback_group = ReentrantCallbackGroup()
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.model_state_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10, callback_group=self.callback_group)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10, callback_group=self.callback_group)
        
        self.current_stage = 1
        self.stage_success_streak = 0
        self.home_x = 2.0
        self.home_y = 2.0
        self.goal_x = 5.0
        self.goal_y = 2.0
        
        self.current_x = 2.0
        self.current_y = 2.0
        self.current_yaw = 0.0
        
        self.min_front_laser = 10.0
        self.min_front_left_laser = 10.0
        self.min_left_laser = 10.0
        self.min_rear_left_laser = 10.0
        self.min_rear_laser = 10.0
        self.min_rear_right_laser = 10.0
        self.min_right_laser = 10.0
        self.min_front_right_laser = 10.0
        
        self.position_history = collections.deque(maxlen=40) 
        
        self.turn_waypoints = [] 
        self.last_yaw = 0.0
        self.episode_max_dist = 0.0
        self.stagnant_episodes_count = 0
        
        self.last_episode_final_x = 0.0
        self.last_episode_final_y = 0.0
        self.consecutive_coordinate_lock_count = 0
        self.backtrack_depth_multiplier = 1
        
        self.force_random_exploration_steps = 0
        
        self.backtracking_mode = False
        self.target_backtrack_waypoint = None
        self.last_spawn_base_x = 2.0
        self.last_spawn_base_y = 2.0
        
        self.rth_mode = False
        
        self.actions = [
            (0.3, 0.0),   
            (0.15, 1.5),  
            (0.15, -1.5), 
            (-0.25, 0.0), 
            (0.0, 0.0)    
        ]
        
        self.memory = collections.deque(maxlen=2500)
        self.model = DQN(10, 5) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.min_epsilon = 0.05
        self.gamma = 0.98
        self.batch_size = 64
        
        self.episode = 1
        self.steps = 0
        self.total_reward = 0.0
        
        self.state = np.zeros(10, dtype=np.float32)
        self.action_idx = 0
        
        self.is_resetting = False
        self.update_curriculum_arena_coordinates()
        self.teleport_reset()
        
        self.timer = self.create_timer(0.1, self.control_loop_callback, callback_group=self.callback_group)
        
    def update_curriculum_arena_coordinates(self):
        if self.current_stage == 1:
            self.home_x = 2.0
            self.home_y = 2.0
            self.goal_x = 5.0
            self.goal_y = 2.0
        elif self.current_stage == 2:
            self.home_x = 22.0
            self.home_y = 2.0
            self.goal_x = 25.0
            self.goal_y = 4.5
        elif self.current_stage == 3:
            self.home_x = 42.0
            self.home_y = 2.0
            self.goal_x = 45.0
            self.goal_y = 1.25

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        ranges = msg.ranges
        num_samples = len(ranges)
        if num_samples == 0:
            return
            
        front_indices = list(range(0, 20)) + list(range(num_samples - 20, num_samples))
        front_left_indices = list(range(45 - 15, 45 + 15))
        left_indices = list(range(90 - 20, 90 + 20))
        rear_left_indices = list(range(135 - 15, 135 + 15))
        rear_indices = list(range(180 - 20, 180 + 20))
        rear_right_indices = list(range(225 - 15, 225 + 15))
        right_indices = list(range(270 - 20, 270 + 20))
        front_right_indices = list(range(315 - 15, 315 + 15))
        
        def get_min_range(indices):
            valid_vals = [ranges[i] for i in indices if i < num_samples and ranges[i] > 0.26 and not np.isinf(ranges[i]) and not np.isnan(ranges[i])]
            return min(valid_vals) if valid_vals else 10.0

        self.min_front_laser = get_min_range(front_indices)
        self.min_front_left_laser = get_min_range(front_left_indices)
        self.min_left_laser = get_min_range(left_indices)
        self.min_rear_left_laser = get_min_range(rear_left_indices)
        self.min_rear_laser = get_min_range(rear_indices)
        self.min_rear_right_laser = get_min_range(rear_right_indices)
        self.min_right_laser = get_min_range(right_indices)
        self.min_front_right_laser = get_min_range(front_right_indices)

    def teleport_reset(self):
        self.is_resetting = True
        self.min_front_laser = 10.0
        self.min_front_left_laser = 10.0
        self.min_left_laser = 10.0
        self.min_rear_left_laser = 10.0
        self.min_rear_laser = 10.0
        self.min_rear_right_laser = 10.0
        self.min_right_laser = 10.0
        self.min_front_right_laser = 10.0
        self.position_history.clear()
        
        twist = Twist()
        self.cmd_pub.publish(twist)
        
        state = ModelState()
        state.model_name = 'tushar_car_v11'
        
        random_yaw = random.uniform(-math.pi, math.pi)
        half_yaw = random_yaw * 0.5
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = math.sin(half_yaw)
        state.pose.orientation.w = math.cos(half_yaw)
        
        jitter_x = 0.0
        jitter_y = 0.0
        if self.consecutive_coordinate_lock_count >= 3:
            jitter_x = random.uniform(-0.1, 0.1)
            jitter_y = random.uniform(-0.1, 0.1)
            self.force_random_exploration_steps = 20 
            print(f"--> [SPAWN ESCAPE MECHANISM] Applying geometric shift: ({jitter_x:.2f}, {jitter_y:.2f})")
        
        if self.backtracking_mode and self.target_backtrack_waypoint:
            print(f"--> [STACK RECOVERY] Relocating back to historic node coordinate: {self.target_backtrack_waypoint}")
            self.last_spawn_base_x = self.target_backtrack_waypoint[0] + jitter_x
            self.last_spawn_base_y = self.target_backtrack_waypoint[1] + jitter_y
            self.backtracking_mode = False
            
            state.pose.position.x = self.last_spawn_base_x
            state.pose.position.y = self.last_spawn_base_y
            state.pose.position.z = 0.15
            state.twist.linear.x = 0.0; state.twist.linear.y = 0.0; state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0; state.twist.angular.y = 0.0; state.twist.angular.z = 0.0
            self.model_state_pub.publish(state)
        elif self.stagnant_episodes_count >= 5:
            if self.turn_waypoints:
                if self.consecutive_coordinate_lock_count >= 2:
                    self.backtrack_depth_multiplier += 1
                    purge_count = min(self.backtrack_depth_multiplier * 2, len(self.turn_waypoints) - 1)
                    print(f"--> [PERSISTENT LOCK DETECTED] LockCount: {self.consecutive_coordinate_lock_count}. Window: {purge_count + 1} nodes deep down stack graph.")
                    for _ in range(purge_count):
                        self.turn_waypoints.pop()
                    self.target_backtrack_waypoint = self.turn_waypoints.pop()
                else:
                    self.target_backtrack_waypoint = self.turn_waypoints.pop()
                    print(f"--> [STACK POP] Standard step backtracking activated. Falling back to: {self.target_backtrack_waypoint}")
                
                self.last_spawn_base_x = self.target_backtrack_waypoint[0] + jitter_x
                self.last_spawn_base_y = self.target_backtrack_waypoint[1] + jitter_y
                
                state.pose.position.x = self.last_spawn_base_x
                state.pose.position.y = self.last_spawn_base_y
                state.pose.position.z = 0.15
                state.twist.linear.x = 0.0; state.twist.linear.y = 0.0; state.twist.linear.z = 0.0
                state.twist.angular.x = 0.0; state.twist.angular.y = 0.0; state.twist.angular.z = 0.0
                self.model_state_pub.publish(state)
            else:
                print("--> [STACK EXHAUSTED] Cascading down to initial Home Base via active RTH driving vector.")
                self.rth_mode = True
                self.last_spawn_base_x = self.home_x + jitter_x
                self.last_spawn_base_y = self.home_y + jitter_y
            self.stagnant_episodes_count = 0
        else:
            self.last_spawn_base_x = self.home_x + jitter_x
            self.last_spawn_base_y = self.home_y + jitter_y
            state.pose.position.x = self.last_spawn_base_x
            state.pose.position.y = self.last_spawn_base_y
            state.pose.position.z = 0.15
            state.twist.linear.x = 0.0; state.twist.linear.y = 0.0; state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0; state.twist.angular.y = 0.0; state.twist.angular.z = 0.0
            self.model_state_pub.publish(state)
            
        self.episode_max_dist = 0.0
        self.is_resetting = False

    def select_action(self, state):
        if self.force_random_exploration_steps > 0:
            self.force_random_exploration_steps -= 1
            return random.randint(0, 4)
            
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)
        
        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)
        next_q = self.model(next_states_t).max(1)[0]
        target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))
        
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def control_loop_callback(self):
        if self.is_resetting:
            return

        if self.rth_mode:
            dx = self.home_x - self.current_x
            dy = self.home_y - self.current_y
            dist_to_home = np.sqrt(dx**2 + dy**2)
            
            if dist_to_home < 0.4:
                print("--> [RTH SUCCESS] Arrived safely at structural initial home base position.")
                self.rth_mode = False
                self.stagnant_episodes_count = 0
                self.consecutive_coordinate_lock_count = 0
                self.backtrack_depth_multiplier = 1
                self.teleport_reset()
                return
                
            twist = Twist()
            if self.min_front_laser < 0.55:
                twist.linear.x = 0.1
                twist.angular.z = 1.5 if self.min_left_laser >= self.min_right_laser else -1.5
            elif self.min_left_laser < 0.38:
                twist.linear.x = 0.2
                twist.angular.z = -1.0
            elif self.min_right_laser < 0.38:
                twist.linear.x = 0.2
                twist.angular.z = 1.0
            else:
                target_angle = math.atan2(dy, dx)
                angle_error = target_angle - self.current_yaw
                while angle_error > math.pi: angle_error -= 2.0 * math.pi
                while angle_error < -math.pi: angle_error += 2.0 * math.pi
                
                if abs(angle_error) > 0.3:
                    twist.linear.x = 0.0
                    twist.angular.z = 1.5 if angle_error > 0 else -1.5
                else:
                    twist.linear.x = 0.3
                    twist.angular.z = 0.5 * angle_error
                    
            self.cmd_pub.publish(twist)
            return

        current_linear_vel = self.actions[self.action_idx][0]
        current_angular_vel = self.actions[self.action_idx][1]

        next_state = np.array([
            self.current_x, 
            self.current_y, 
            current_linear_vel,
            current_angular_vel,
            self.min_front_laser, 
            self.min_left_laser, 
            self.min_rear_laser, 
            self.min_right_laser,
            min(self.min_front_left_laser, self.min_rear_left_laser),
            min(self.min_front_right_laser, self.min_rear_right_laser)
        ], dtype=np.float32)
        
        dist_to_goal = np.sqrt((self.current_x - self.goal_x)**2 + (self.current_y - self.goal_y)**2)
        dist_from_spawn = np.sqrt((self.current_x - self.last_spawn_base_x)**2 + (self.current_y - self.last_spawn_base_y)**2)
        
        if dist_from_spawn > self.episode_max_dist:
            self.episode_max_dist = dist_from_spawn
            
        if abs(self.current_yaw - self.last_yaw) > 0.52:
            if self.min_front_laser > 0.8 and self.min_rear_laser > 0.8:
                if not self.turn_waypoints or np.sqrt((self.current_x - self.turn_waypoints[-1][0])**2 + (self.current_y - self.turn_waypoints[-1][1])**2) > 0.8:
                    self.turn_waypoints.append((self.current_x, self.current_y))
                    if len(self.turn_waypoints) > 15: 
                        self.turn_waypoints.pop(0)
        self.last_yaw = self.current_yaw

        reward = -dist_to_goal * 0.2
        
        all_laser_sectors = [
            self.min_front_laser, self.min_left_laser, self.min_rear_laser, self.min_right_laser,
            self.min_front_left_laser, self.min_rear_left_laser, self.min_front_right_laser, self.min_rear_right_laser
        ]
        closest_obstacle_dist = min(all_laser_sectors)
        
        if closest_obstacle_dist < 0.7:
            reward -= 15.0 * (0.7 - closest_obstacle_dist)
            if (self.min_front_laser < 0.6 or self.min_front_left_laser < 0.55 or self.min_front_right_laser < 0.55) and self.action_idx == 3:
                reward += 20.0

        if self.min_front_laser < 1.3:
            if self.action_idx == 0: 
                reward -= 50.0  
            elif self.min_left_laser > self.min_front_laser and self.action_idx == 1: 
                reward += 15.0
            elif self.min_right_laser > self.min_front_laser and self.action_idx == 2: 
                reward += 15.0

        done = False
        
        self.position_history.append((self.current_x, self.current_y))
        
        if len(self.position_history) == self.position_history.maxlen:
            old_x, old_y = self.position_history[0]
            net_displacement = np.sqrt((self.current_x - old_x)**2 + (self.current_y - old_y)**2)
            
            if net_displacement < 0.05 and self.action_idx != 4:
                print(f"--> [MESH FRICTION LOCK DETECTED AT ({self.current_x:.2f}, {self.current_y:.2f})]")
                
                escape_x = self.current_x - 0.25 * math.cos(self.current_yaw)
                escape_y = self.current_y - 0.25 * math.sin(self.current_yaw)
                
                state_msg = ModelState()
                state_msg.model_name = 'tushar_car_v11'
                state_msg.pose.position.x = escape_x
                state_msg.pose.position.y = escape_y
                state_msg.pose.position.z = 0.15
                
                half_yaw = self.current_yaw * 0.5
                state_msg.pose.orientation.x = 0.0
                state_msg.pose.orientation.y = 0.0
                state_msg.pose.orientation.z = math.sin(half_yaw)
                state_msg.pose.orientation.w = math.cos(half_yaw)
                
                state_msg.twist.linear.x = 0.0; state_msg.twist.linear.y = 0.0; state_msg.twist.linear.z = 0.0
                state_msg.twist.angular.x = 0.0; state_msg.twist.angular.y = 0.0; state_msg.twist.angular.z = 0.0
                
                self.model_state_pub.publish(state_msg)
                
                self.position_history.clear()
                reward -= 15.0
                self.force_random_exploration_steps = 8 

        is_immediate_spawn_crash = False
        if self.steps > 0:
            if dist_to_goal < 0.4:
                reward = 300.0
                done = True
                self.stage_success_streak += 1
            elif self.min_front_laser < 0.46 or self.min_front_left_laser < 0.42 or self.min_front_right_laser < 0.42:  
                reward = -5.0 if self.steps < 5 else -70.0
                if self.steps < 5: is_immediate_spawn_crash = True
                done = True
                self.stage_success_streak = 0
            elif self.min_rear_laser < 0.38 or self.min_rear_left_laser < 0.38 or self.min_rear_right_laser < 0.38:
                reward = -5.0 if self.steps < 5 else -70.0
                if self.steps < 5: is_immediate_spawn_crash = True
                done = True
                self.stage_success_streak = 0
            elif (self.current_stage == 1 and (self.current_x < 0.3 or self.current_y < 0.3 or self.current_x > 7.7 or self.current_y > 7.7)):
                reward = -70.0
                done = True
                self.stage_success_streak = 0
            elif (self.current_stage == 2 and (self.current_x < 20.3 or self.current_y < 0.3 or self.current_x > 27.7 or self.current_y > 7.7)):
                reward = -70.0
                done = True
                self.stage_success_streak = 0
            elif (self.current_stage == 3 and (self.current_x < 40.3 or self.current_y < 0.3 or self.current_x > 47.7 or self.current_y > 7.7)):
                reward = -70.0
                done = True
                self.stage_success_streak = 0
            elif self.steps >= 300:
                done = True
                self.stage_success_streak = 0

            if not is_immediate_spawn_crash:
                self.memory.append((self.state, self.action_idx, reward, next_state, done))
            self.total_reward += reward
            self.train_step()

        if done:
            if self.steps < 15 and dist_to_goal > 1.0:
                print(f"--> [CRASH STUDY] Short episode detected ({self.steps} steps). Forcing 5x intense training updates...")
                for _ in range(5):
                    self.train_step()

            if self.stage_success_streak >= 5 and self.current_stage < 3:
                self.current_stage += 1
                self.stage_success_streak = 0
                self.turn_waypoints.clear()
                self.update_curriculum_arena_coordinates()
                print(f"\n========================================================")
                print(f"--> [CURRICULUM GRADUATION] Graduated to Arena Stage {self.current_stage}!")
                print(f"--> Teleporting to Home Offset: ({self.home_x}, {self.home_y})")
                print(f"========================================================\n")

            err_from_prev_failure_x = abs(self.current_x - self.last_episode_final_x)
            err_from_prev_failure_y = abs(self.current_y - self.last_episode_final_y)
            
            if err_from_prev_failure_x < 0.35 and err_from_prev_failure_y < 0.35:
                self.consecutive_coordinate_lock_count += 1
            else:
                self.consecutive_coordinate_lock_count = 0
                self.backtrack_depth_multiplier = 1
                
            self.last_episode_final_x = self.current_x
            self.last_episode_final_y = self.current_y

            if self.episode_max_dist < 1.2:
                self.stagnant_episodes_count += 1
            else:
                self.stagnant_episodes_count = 0 
                
            if self.stagnant_episodes_count >= 5:
                if self.turn_waypoints:
                    if self.consecutive_coordinate_lock_count >= 2:
                        self.backtrack_depth_multiplier += 1
                        purge_count = min(self.backtrack_depth_multiplier * 2, len(self.turn_waypoints) - 1)
                        print(f"--> [LOOP ESCAPE PURGE] Dropping {purge_count + 1} turns back down history stack.")
                        for _ in range(purge_count):
                            self.turn_waypoints.pop()
                        self.target_backtrack_waypoint = self.turn_waypoints.pop()
                    else:
                        self.target_backtrack_waypoint = self.turn_waypoints.pop()
                    self.backtracking_mode = True
                else:
                    print("--> [STACK EMPTY] Overriding loops to force physical RTH sequence.")
                    self.rth_mode = True
                    self.stagnant_episodes_count = 0
                    self.consecutive_coordinate_lock_count = 0
                    self.backtrack_depth_multiplier = 1
                    self.episode += 1
                    self.steps = 0
                    self.total_reward = 0.0
                    return

            print(f"Stage: {self.current_stage} (Streak: {self.stage_success_streak}) | Episode: {self.episode} | Score: {self.total_reward:.2f} | LockCount: {self.consecutive_coordinate_lock_count} | End Pos: ({self.current_x:.2f}, {self.current_y:.2f})")
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.episode += 1
            self.steps = 0
            self.total_reward = 0.0
            self.teleport_reset()
            return

        self.state = next_state
        
        self.action_idx = self.select_action(self.state)
        linear, angular = self.actions[self.action_idx]
            
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)
        self.steps += 1

def main():
    rclpy.init()
    node = GazeboQLearningNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()