#!/home/dat/env/bin/python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from dynamic_obstacles.msg import DynamicObstacles, DynamicObstacle
from geometry_msgs.msg import Point, Vector3, PointStamped
import message_filters
import tf2_ros
import tf2_geometry_msgs
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter
from collections import defaultdict

class DynamicObstacleProcessor:
    def __init__(self):
        rospy.init_node('dynamic_obstacle_processor', anonymous=True)

        # Parameters
        self.robot_ids = ['robot_1', 'robot_2', 'robot_3']
        self.cluster_eps = rospy.get_param('~cluster_eps', 0.2)  # DBSCAN clustering distance (m)
        self.min_points = rospy.get_param('~min_points', 5)  # Minimum points per cluster
        self.max_distance = rospy.get_param('~max_distance', 5.0)  # Max LiDAR detection range (m)
        self.cluster_size_min = rospy.get_param('~cluster_size_min', 0.3)  # Min human diameter (m)
        self.cluster_size_max = rospy.get_param('~cluster_size_max', 0.6)  # Max human diameter (m)
        self.merge_threshold = rospy.get_param('~merge_threshold', 0.5)  # Distance to merge clusters (m)
        self.human_radius = rospy.get_param('~human_radius', 0.3)  # Human radius (m)

        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.pub = rospy.Publisher('/dynamic_obstacles', DynamicObstacles, queue_size=10)

        # Subscribers for each robot (synchronized scan and odometry)
        self.subscribers = []
        for robot_id in self.robot_ids:
            scan_sub = message_filters.Subscriber(f'/{robot_id}/scan', LaserScan)
            odom_sub = message_filters.Subscriber(f'/{robot_id}/odom', Odometry)
            ts = message_filters.ApproximateTimeSynchronizer(
                [scan_sub, odom_sub], queue_size=10, slop=0.1)
            ts.registerCallback(lambda scan, odom, rid=robot_id: self.lidar_callback(scan, odom, rid))
            self.subscribers.append(ts)

        # Storage for detected obstacles and Kalman filters
        self.detected_obstacles = defaultdict(list)
        self.kalman_filters = {}
        self.next_id = 0

    def initialize_kalman(self, position):
        """Initialize a Kalman filter for a new obstacle."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, v_x, v_y], Measurement: [x, y]
        kf.x = np.array([position[0], position[1], 0.0, 0.0])  # Initial velocity = 0
        dt = 0.2  # 5Hz update rate
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 10.0  # Initial uncertainty
        kf.R = np.eye(2) * 0.1  # Measurement noise
        kf.Q = np.eye(4) * 0.01  # Process noise
        return kf

    def lidar_callback(self, scan_msg, odom_msg, robot_id):
        """Process LiDAR data and odometry for a single robot."""
        # Convert LiDAR ranges to Cartesian points
        points = []
        for i, r in enumerate(scan_msg.ranges):
            if scan_msg.range_min < r < self.max_distance:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])

        if not points:
            return

        points = np.array(points)

        # Cluster points using DBSCAN
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.min_points, n_jobs=-1).fit(points)
        labels = db.labels_

        # Process clusters
        clusters = []
        for label in set(labels) - {-1}:  # Ignore noise
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            cluster_size = np.max(np.linalg.norm(cluster_points - centroid, axis=1)) * 2
            if self.cluster_size_min <= cluster_size <= self.cluster_size_max:
                clusters.append(centroid)

        # Transform clusters to world frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', f'{robot_id}/base_link', scan_msg.header.stamp, rospy.Duration(0.1))
            transformed_clusters = []
            for centroid in clusters:
                point = Point(x=centroid[0], y=centroid[1], z=0.0)
                point_stamped = PointStamped(header=scan_msg.header, point=point)
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                transformed_clusters.append([transformed_point.point.x, transformed_point.point.y])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF error for {robot_id}: {e}")
            return

        # Store clusters
        self.detected_obstacles[robot_id] = transformed_clusters

        # Process and publish
        self.process_and_publish()

    def process_and_publish(self):
        """Merge clusters from all robots and publish dynamic obstacles."""
        # Collect all clusters
        all_clusters = []
        for robot_id in self.robot_ids:
            all_clusters.extend(self.detected_obstacles[robot_id])

        if not all_clusters:
            return

        # Merge duplicate clusters
        clusters = np.array(all_clusters)
        if len(clusters) > 1:
            db = DBSCAN(eps=self.merge_threshold, min_samples=1, n_jobs=-1).fit(clusters)
            merged_clusters = []
            for label in set(db.labels_):
                cluster_points = clusters[db.labels_ == label]
                merged_clusters.append(np.mean(cluster_points, axis=0))
        else:
            merged_clusters = clusters

        # Update Kalman filters and create obstacles
        obstacles_msg = DynamicObstacles()
        obstacles_msg.header.stamp = rospy.Time.now()
        obstacles_msg.header.frame_id = 'world'

        current_obstacles = []
        for centroid in merged_clusters:
            # Find matching Kalman filter
            min_dist = float('inf')
            matched_id = None
            for oid, kf in self.kalman_filters.items():
                dist = np.linalg.norm(centroid - kf.x[:2])
                if dist < min_dist and dist < self.merge_threshold:
                    min_dist = dist
                    matched_id = oid

            if matched_id is None:
                # New obstacle
                kf = self.initialize_kalman(centroid)
                oid = self.next_id
                self.kalman_filters[oid] = kf
                self.next_id += 1
            else:
                kf = self.kalman_filters[matched_id]

            # Update Kalman filter
            kf.predict()
            kf.update(centroid)

            # Create obstacle
            obstacle = DynamicObstacle()
            obstacle.id = oid
            obstacle.position = Point(x=kf.x[0], y=kf.x[1], z=0.0)
            obstacle.velocity = Vector3(x=kf.x[2], y=kf.x[3], z=0.0)
            obstacle.radius = self.human_radius
            current_obstacles.append(obstacle)

        obstacles_msg.obstacles = current_obstacles

        # Clean up unused Kalman filters
        used_ids = {obs.id for obs in current_obstacles}
        self.kalman_filters = {oid: kf for oid, kf in self.kalman_filters.items() if oid in used_ids}

        # Publish
        self.pub.publish(obstacles_msg)
        rospy.loginfo(f"Published {len(current_obstacles)} dynamic obstacles")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        processor = DynamicObstacleProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")