import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_OK = True
except ImportError:
    CV_BRIDGE_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

ANIMATE_IDS = {0, 15, 16, 17, 18, 19, 20, 21, 22, 23}
INANIMATE_IDS = {24, 25, 26, 28, 32, 56, 57, 58, 59, 60, 63, 64, 67}
CONF_THRESHOLD = 0.45
CAMERA_FOV_H = 1.3962634
IMG_WIDTH = 320
CAMERA_OFFSET_X = 0.19

class ObstacleClassifierNode(Node):
    def __init__(self):
        super().__init__('obstacle_classifier')
        self.declare_parameter('robot_name', 'robot_1')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.get_logger().info(f'ObstacleClassifier starting for [{self.robot_name}]')
        self.model = None
        if YOLO_OK:
            try:
                self.model = YOLO('yolov8n.pt')
                self.get_logger().info('YOLOv8n loaded.')
            except Exception as e:
                self.get_logger().warn(f'YOLO load failed: {e}')
        else:
            self.get_logger().warn('ultralytics not installed. No detections.')
        self.bridge = CvBridge() if CV_BRIDGE_OK else None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.lidar_ranges = []
        self.lidar_angle_min = 0.0
        self.lidar_angle_inc = 0.0
        self.create_subscription(Image, f'/{self.robot_name}/camera/image_raw', self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.lidar_callback, qos_profile_sensor_data)
        self.detection_pub = self.create_publisher(MarkerArray, f'/{self.robot_name}/obstacle_detections', 10)
        self.get_logger().info(f'[{self.robot_name}] ObstacleClassifier ready.')

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.pose_x = pos.x
        self.pose_y = pos.y
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.pose_yaw = math.atan2(siny, cosy)

    def lidar_callback(self, msg):
        self.lidar_ranges = list(msg.ranges)
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_inc = msg.angle_increment

    def image_callback(self, msg):
        if self.model is None or self.bridge is None:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model(cv_img, verbose=False, conf=CONF_THRESHOLD)
        except Exception as e:
            self.get_logger().warn(f'inference failed: {e}')
            return
        ma = MarkerArray()
        marker_id = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id not in ANIMATE_IDS and cls_id not in INANIMATE_IDS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx_px = (x1 + x2) / 2.0
                angle_in_cam = ((cx_px / IMG_WIDTH) - 0.5) * CAMERA_FOV_H
                angle_world = self.pose_yaw + angle_in_cam
                depth = self._lidar_depth_at_angle(angle_in_cam)
                if depth is None or depth > 8.0:
                    continue
                cam_wx = self.pose_x + CAMERA_OFFSET_X * math.cos(self.pose_yaw)
                cam_wy = self.pose_y + CAMERA_OFFSET_X * math.sin(self.pose_yaw)
                obs_x = cam_wx + depth * math.cos(angle_world)
                obs_y = cam_wy + depth * math.sin(angle_world)
                m = Marker()
                m.header.stamp = msg.header.stamp
                m.header.frame_id = 'map'
                m.ns = f'{self.robot_name}_yolo'
                m.id = marker_id
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.pose.position.x = obs_x
                m.pose.position.y = obs_y
                m.pose.position.z = 1.2
                m.pose.orientation.w = 1.0
                m.scale.z = 0.3
                m.text = self.model.names[cls_id]
                m.color.r = 1.0
                m.color.g = 0.0 if cls_id in ANIMATE_IDS else 0.5
                m.color.b = 0.0
                m.color.a = 1.0
                m.lifetime.sec = 1
                ma.markers.append(m)
                marker_id += 1
        self.detection_pub.publish(ma)

    def _lidar_depth_at_angle(self, angle_cam):
        if not self.lidar_ranges:
            return None
        best_dist = None
        search_half = math.radians(5.0)
        for i, r in enumerate(self.lidar_ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0.05:
                continue
            ray_angle = self.lidar_angle_min + i * self.lidar_angle_inc
            diff = ray_angle - angle_cam
            while diff > math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            if abs(diff) <= search_half:
                if best_dist is None or r < best_dist:
                    best_dist = r
        return best_dist

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
