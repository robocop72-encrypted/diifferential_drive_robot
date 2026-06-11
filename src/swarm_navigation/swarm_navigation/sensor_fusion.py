import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

MAP_RESOLUTION = 0.05
MAP_WIDTH_M = 30.0
MAP_HEIGHT_M = 30.0
MAP_ORIGIN_X = -2.0
MAP_ORIGIN_Y = -2.0
LIDAR_HIT_INC = 15
LIDAR_FREE_DEC = 4
OCC_MAX = 100
OCC_MIN = 0
OCC_THRESHOLD = 50
DECAY_RATE = 0.995

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        self.declare_parameter('robot_name', 'robot_1')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.get_logger().info(f'SensorFusion starting for [{self.robot_name}]')
        self.map_cols = int(MAP_WIDTH_M / MAP_RESOLUTION)
        self.map_rows = int(MAP_HEIGHT_M / MAP_RESOLUTION)
        self.grid = np.full((self.map_rows, self.map_cols), -1.0, dtype=np.float32)
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.dynamic_obstacles = []
        reliable_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.lidar_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        self.create_subscription(MarkerArray, f'/{self.robot_name}/obstacle_detections', self.detection_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, f'/{self.robot_name}/fused_map', reliable_qos)
        self.scan_pub = self.create_publisher(LaserScan, f'/{self.robot_name}/fused_scan', qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, f'/{self.robot_name}/fusion_markers', 10)
        self.create_timer(0.5, self.publish_map)
        self.get_logger().info(f'[{self.robot_name}] SensorFusion ready.')

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.pose_x = pos.x
        self.pose_y = pos.y
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.pose_yaw = math.atan2(siny, cosy)

    def lidar_callback(self, msg):
        rx, ry, ryaw = self.pose_x, self.pose_y, self.pose_yaw
        mask = self.grid > 0
        self.grid[mask] *= DECAY_RATE
        angle = msg.angle_min
        for r in msg.ranges:
            angle += msg.angle_increment
            if math.isnan(r) or math.isinf(r) or r < msg.range_min or r > msg.range_max:
                continue
            ray_angle_world = angle + ryaw
            hit_x = rx + r * math.cos(ray_angle_world)
            hit_y = ry + r * math.sin(ray_angle_world)
            self._bresenham_free(rx, ry, hit_x, hit_y)
            if r < msg.range_max - 0.05:
                self._mark_occupied(hit_x, hit_y)
        clean = LaserScan()
        clean.header = msg.header
        clean.header.frame_id = f'{self.robot_name}/laser_link'
        clean.angle_min = msg.angle_min
        clean.angle_max = msg.angle_max
        clean.angle_increment = msg.angle_increment
        clean.time_increment = msg.time_increment
        clean.scan_time = msg.scan_time
        clean.range_min = msg.range_min
        clean.range_max = msg.range_max
        clean.ranges = msg.ranges
        clean.intensities = msg.intensities
        self.scan_pub.publish(clean)

    def detection_callback(self, msg):
        self.dynamic_obstacles.clear()
        for m in msg.markers:
            wx, wy = m.pose.position.x, m.pose.position.y
            self._inflate_obstacle(wx, wy, radius_m=0.3)
            self.dynamic_obstacles.append((wx, wy, 'detected'))

    def _world_to_grid(self, wx, wy):
        col = int((wx - MAP_ORIGIN_X) / MAP_RESOLUTION)
        row = int((wy - MAP_ORIGIN_Y) / MAP_RESOLUTION)
        if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
            return col, row
        return None

    def _mark_occupied(self, wx, wy):
        cell = self._world_to_grid(wx, wy)
        if cell:
            col, row = cell
            self.grid[row, col] = min(OCC_MAX, self.grid[row, col] + LIDAR_HIT_INC if self.grid[row, col] >= 0 else LIDAR_HIT_INC)

    def _bresenham_free(self, x0, y0, x1, y1):
        c0 = self._world_to_grid(x0, y0)
        c1 = self._world_to_grid(x1, y1)
        if c0 is None or c1 is None:
            return
        col0, row0 = c0
        col1, row1 = c1
        dc = abs(col1 - col0)
        dr = abs(row1 - row0)
        sc = 1 if col1 > col0 else -1
        sr = 1 if row1 > row0 else -1
        err = dc - dr
        steps = max(dc, dr)
        col, row = col0, row0
        for _ in range(steps - 1):
            if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
                self.grid[row, col] = max(OCC_MIN, self.grid[row, col] - LIDAR_FREE_DEC if self.grid[row, col] >= 0 else 0)
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                col += sc
            if e2 < dc:
                err += dc
                row += sr

    def _inflate_obstacle(self, wx, wy, radius_m=0.3):
        r_cells = int(radius_m / MAP_RESOLUTION) + 1
        centre = self._world_to_grid(wx, wy)
        if centre is None:
            return
        cc, cr = centre
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    row, col = cr + dr, cc + dc
                    if 0 <= row < self.map_rows and 0 <= col < self.map_cols:
                        self.grid[row, col] = float(OCC_MAX)

    def publish_map(self):
        now = self.get_clock().now().to_msg()
        occ = OccupancyGrid()
        occ.header.stamp = now
        occ.header.frame_id = 'map'
        occ.info.resolution = MAP_RESOLUTION
        occ.info.width = self.map_cols
        occ.info.height = self.map_rows
        occ.info.origin = Pose()
        occ.info.origin.position.x = MAP_ORIGIN_X
        occ.info.origin.position.y = MAP_ORIGIN_Y
        occ.info.origin.orientation.w = 1.0
        flat = self.grid.flatten()
        data = []
        for v in flat:
            if v < 0:
                data.append(-1)
            elif v >= OCC_THRESHOLD:
                data.append(100)
            else:
                data.append(int(v))
        occ.data = data
        self.map_pub.publish(occ)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
