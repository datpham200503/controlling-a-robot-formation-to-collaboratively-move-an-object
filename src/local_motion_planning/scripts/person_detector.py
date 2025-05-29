#!/home/dat/env/bin/python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from sklearn.cluster import DBSCAN
import tf2_ros
import tf2_geometry_msgs
import message_filters
from filterpy.kalman import KalmanFilter
import math
import uuid

class PersonDetector:
    def __init__(self):
        # Khởi tạo node ROS
        rospy.init_node('person_detector', anonymous=True)
        
        # Subscriber cho ba topic /scan
        scan_sub1 = message_filters.Subscriber('/robot_1/scan', LaserScan)
        scan_sub2 = message_filters.Subscriber('/robot_2/scan', LaserScan)
        scan_sub3 = message_filters.Subscriber('/robot_3/scan', LaserScan)
        
        # Đồng bộ ba topic
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [scan_sub1, scan_sub2, scan_sub3], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.scan_callback)
        
        # Publisher cho marker
        self.marker_pub = rospy.Publisher('/person_markers', MarkerArray, queue_size=10)
        
        # TF2 buffer và listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Tham số cho DBSCAN
        self.eps = 0.3  # Bán kính cụm (m)
        self.min_samples = 5  # Số điểm tối thiểu trong cụm
        self.min_size = 0.3  # Kích thước tối thiểu của cụm (m)
        self.max_size = 1.0  # Kích thước tối đa của cụm (m)
        self.min_velocity = 0.05  # Vận tốc tối thiểu (m/s)
        self.cluster_timeout = rospy.Duration(2.0)  # Thời gian hết hạn cụm
        
        # Lưu trữ lịch sử cụm
        self.clusters = {}  # {cluster_id: {'kalman': KalmanFilter, 'center': (x, y), 'size': size, 'timestamp': rospy.Time}}
        self.id_counter = 0  # Đếm ID nếu không dùng UUID

    def init_kalman(self):
        """Khởi tạo bộ lọc Kalman cho cụm mới."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0 / 7.0  # Tần số 7 Hz
        # Ma trận trạng thái: [x, y, vx, vy]
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        # Ma trận đo lường: đo [x, y]
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        # Hiệp phương sai nhiễu quá trình
        kf.Q = np.eye(4) * 0.1  # Nhiễu nhỏ, giả định chuyển động đều
        # Hiệp phương sai nhiễu đo lường
        kf.R = np.eye(2) * 0.05  # Nhiễu LiDAR ~5cm
        # Hiệp phương sai ban đầu
        kf.P = np.eye(4) * 1.0
        return kf

    def transform_point(self, x, y, source_frame, target_frame, stamp):
        """Chuyển đổi một điểm từ source_frame sang target_frame."""
        point = Point(x=x, y=y, z=0.0)
        point_stamped = tf2_geometry_msgs.PointStamped()
        point_stamped.header.frame_id = source_frame
        point_stamped.header.stamp = stamp
        point_stamped.point = point
        
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(1.0))
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed_point.point.x, transformed_point.point.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF transform failed: {e}")
            return None, None

    def get_robot_positions(self, stamp):
        """Lấy vị trí ba robot trong khung map."""
        robot_frames = ['robot_1/base_link', 'robot_2/base_link', 'robot_3/base_link']
        positions = []
        for frame in robot_frames:
            try:
                transform = self.tf_buffer.lookup_transform('map', frame, stamp, rospy.Duration(1.0))
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                positions.append((x, y))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"Failed to get position of {frame}: {e}")
                positions.append(None)
        return positions

    def filter_robot_points(self, points, robot_positions):
        """Loại bỏ điểm nằm trong vùng 0.4m x 0.4m quanh tâm robot."""
        filtered_points = []
        exclusion_size = 0.4  # Vùng loại trừ 0.4m x 0.4m
        for point in points:
            keep = True
            for pos in robot_positions:
                if pos is None:
                    continue
                x, y = pos
                if (abs(point[0] - x) < exclusion_size / 2 and
                    abs(point[1] - y) < exclusion_size / 2):
                    keep = False
                    break
            if keep:
                filtered_points.append(point)
        return np.array(filtered_points)

    def scan_callback(self, scan1, scan2, scan3):
        """Callback xử lý dữ liệu từ ba LiDAR."""
        scans = [scan1, scan2, scan3]
        scan_frames = ['robot_1/base_scan', 'robot_2/base_scan', 'robot_3/base_scan']
        all_points = []
        
        # Lấy vị trí robot để loại bỏ điểm
        robot_positions = self.get_robot_positions(scan1.header.stamp)
        
        # Xử lý từng scan
        for scan, frame in zip(scans, scan_frames):
            ranges = np.array(scan.ranges)
            angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
            valid_indices = np.isfinite(ranges)
            ranges = ranges[valid_indices]
            angles = angles[valid_indices]
            
            if len(ranges) < self.min_samples:
                rospy.logwarn(f"Not enough valid points in {frame}")
                continue
                
            # Chuyển sang tọa độ Descartes trong khung laser
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)
            points = np.vstack((x, y)).T
            
            # Chuyển sang khung map
            transformed_points = []
            for point in points:
                x_map, y_map = self.transform_point(point[0], point[1], frame, 'map', scan.header.stamp)
                if x_map is not None and y_map is not None:
                    transformed_points.append([x_map, y_map])
            if transformed_points:
                all_points.extend(transformed_points)
        
        # Kiểm tra nếu không có điểm hợp lệ
        all_points = np.array(all_points)
        if len(all_points) < self.min_samples:
            rospy.logwarn("Not enough valid points after transformation")
            return
            
        # Loại bỏ điểm của robot
        all_points = self.filter_robot_points(all_points, robot_positions)
        if len(all_points) < self.min_samples:
            rospy.logwarn("Not enough points after filtering robots")
            return
        
        # Phân cụm với DBSCAN
        try:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(all_points)
            labels = clustering.labels_
        except Exception as e:
            rospy.logerr(f"DBSCAN clustering failed: {e}")
            return
        
        if len(np.unique(labels)) <= 1 and -1 in labels:
            rospy.logwarn("No valid clusters found")
            return
        
        # Tạo MarkerArray
        marker_array = MarkerArray()
        marker_id = 0
        
        # Gán ID và cập nhật Kalman
        new_clusters = {}
        current_time = scan1.header.stamp
        for label in set(labels):
            if label == -1:  # Bỏ qua nhiễu
                continue
                
            try:
                cluster_points = all_points[labels == label]
            except TypeError as e:
                rospy.logerr(f"Error accessing cluster points: {e}")
                continue
                
            if len(cluster_points) < self.min_samples:
                continue
                
            # Tính tâm và kích thước cụm
            min_x, min_y = np.min(cluster_points, axis=0)
            max_x, max_y = np.max(cluster_points, axis=0)
            size_x = max_x - min_x
            size_y = max_y - min_y
            size = max(size_x, size_y)
            
            if self.min_size <= size <= self.max_size:
                center_x, center_y = np.mean(cluster_points, axis=0)
                
                # Gán ID cụm
                cluster_id = None
                min_dist = float('inf')
                for cid, data in self.clusters.items():
                    dist = np.sqrt((center_x - data['center'][0])**2 + (center_y - data['center'][1])**2)
                    if dist < 0.5 and dist < min_dist:
                        min_dist = dist
                        cluster_id = cid
                
                if cluster_id is None:
                    cluster_id = str(self.id_counter)
                    self.id_counter += 1
                    new_clusters[cluster_id] = {
                        'kalman': self.init_kalman(),
                        'center': (center_x, center_y),
                        'size': size,
                        'timestamp': current_time
                    }
                    new_clusters[cluster_id]['kalman'].x = np.array([center_x, center_y, 0.0, 0.0])
                else:
                    new_clusters[cluster_id] = self.clusters[cluster_id]
                    new_clusters[cluster_id]['center'] = (center_x, center_y)
                    new_clusters[cluster_id]['size'] = size
                    new_clusters[cluster_id]['timestamp'] = current_time
                
                # Cập nhật Kalman
                kf = new_clusters[cluster_id]['kalman']
                kf.update(np.array([center_x, center_y]))
                kf.predict()
                
                # Kiểm tra vận tốc
                vx, vy = kf.x[2], kf.x[3]
                velocity = np.sqrt(vx**2 + vy**2)
                if velocity < self.min_velocity:
                    continue
                
                # Tính bốn đỉnh cho vị trí hiện tại
                half_size = size / 2.0
                square_min_x = center_x - half_size
                square_max_x = center_x + half_size
                square_min_y = center_y - half_size
                square_max_y = center_y + half_size
                
                vertices_x = [square_min_x, square_max_x, square_max_x, square_min_x]
                vertices_y = [square_min_y, square_min_y, square_max_y, square_max_y]
                
                rospy.loginfo(f"Person detected at cluster {cluster_id} (current, map frame):")
                rospy.loginfo(f"Vertices: [{vertices_x}, {vertices_y}]")
                
                # Tạo marker cho vị trí hiện tại
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = current_time
                marker.ns = "person_current"
                marker.id = marker_id
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                points_list = [
                    Point(square_min_x, square_min_y, 0.0),
                    Point(square_max_x, square_min_y, 0.0),
                    Point(square_max_x, square_max_y, 0.0),
                    Point(square_min_x, square_max_y, 0.0)
                ]
                marker.points = [
                    points_list[0], points_list[1],
                    points_list[1], points_list[2],
                    points_list[2], points_list[3],
                    points_list[3], points_list[0]
                ]
                marker.scale.x = 0.05
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                marker_id += 1
                
                # Dự đoán vị trí sau 2 giây
                pred_x = center_x + vx * 2.0
                pred_y = center_y + vy * 2.0
                pred_min_x = pred_x - half_size
                pred_max_x = pred_x + half_size
                pred_min_y = pred_y - half_size
                pred_max_y = pred_y + half_size
                
                pred_vertices_x = [pred_min_x, pred_max_x, pred_max_x, pred_min_x]
                pred_vertices_y = [pred_min_y, pred_min_y, pred_max_y, pred_max_y]
                
                rospy.loginfo(f"Person predicted at cluster {cluster_id} (2s ahead, map frame):")
                rospy.loginfo(f"Vertices: [{pred_vertices_x}, {pred_vertices_y}]")
                
                # Tạo marker cho vị trí dự đoán
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = current_time
                marker.ns = "person_predicted"
                marker.id = marker_id
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                points_list = [
                    Point(pred_min_x, pred_min_y, 0.0),
                    Point(pred_max_x, pred_min_y, 0.0),
                    Point(pred_max_x, pred_max_y, 0.0),
                    Point(pred_min_x, pred_max_y, 0.0)
                ]
                marker.points = [
                    points_list[0], points_list[1],
                    points_list[1], points_list[2],
                    points_list[2], points_list[3],
                    points_list[3], points_list[0]
                ]
                marker.scale.x = 0.05
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                marker_id += 1
        
        # Cập nhật lịch sử cụm
        self.clusters = {cid: data for cid, data in new_clusters.items()
                         if (current_time - data['timestamp']) < self.cluster_timeout}
        
        # Publish MarkerArray
        self.marker_pub.publish(marker_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = PersonDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass