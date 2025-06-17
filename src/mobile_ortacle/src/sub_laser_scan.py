import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import threading
from filterpy.kalman import KalmanFilter
from matplotlib.patches import Polygon
from std_msgs.msg import Float64MultiArray

class LaserScanProcessor:
    def __init__(self):

        self.fig = plt.figure(figsize=(8, 8))
        plt.ion()  
        

        rospy.init_node('laserscan_processor', anonymous=True)
        

        self.subscriber = rospy.Subscriber('/scan_filtered', LaserScan, self.laserscan_callback)
        

        self.dynamic_obstacles_pub = rospy.Publisher('/dynamic_obstacle', Float64MultiArray, queue_size=10)
        

        self.points = None
        self.timestamp = None
        self.last_processed_timestamp = None
        

        self.points_lock = threading.Lock()
        

        self.eps = rospy.get_param('~eps', 0.2)
        self.min_samples = rospy.get_param('~min_samples', 10)
        self.velocity_threshold = rospy.get_param('~velocity_threshold', 0.001)
        self.max_match_distance = rospy.get_param('~max_match_distance', 0.7)
        self.angle_std_threshold = np.deg2rad(rospy.get_param('~angle_std_threshold_deg', 50))
        self.min_moving_points_ratio = rospy.get_param('~min_moving_points_ratio', 0.5)
        self.prediction_time = rospy.get_param('~prediction_time', 1.0)
        self.point_count_variation_threshold = rospy.get_param('~point_count_variation_threshold', 0.15)  
        self.static_velocity_threshold = rospy.get_param('~static_velocity_threshold', 0.005)
        self.angular_resolution = rospy.get_param('~angular_resolution', np.deg2rad(1))
        

        self.prev_clusters = []
        self.next_cluster_id = 0
        self.kalman_filters = {}
        self.cluster_age = {}
        self.max_age = 25 
        self.cluster_point_counts = {}
        self.dynamic_object_ids = set()
        self.cluster_velocity_history = {} 
        self.L = 3.7
        self.rate = rospy.Rate(8)

    def initialize_kalman_filter(self, centroid, dt):

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([centroid[0], centroid[1], 0, 0])
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1.0
        kf.R = np.array([[0.3, 0], [0, 0.3]])
        kf.Q = np.eye(4) * 0.025
        return kf

    def predict_future_state(self, kf, prediction_time, dt):

        k = int(prediction_time / dt)
        if k <= 0:
            return kf.x[:2]
        future_state = kf.x.copy()
        F = kf.F.copy()
        for _ in range(k):
            future_state = np.dot(F, future_state)
        return future_state[:2]

    def compute_convex_hull_rectangle(self, points, future_pos):

        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        width = x_max - x_min
        height = y_max - y_min
        
        width *= 2.0
        height *= 2.0
        
        x_left = future_pos[0] - width / 2.0
        x_right = future_pos[0] + width / 2.0
        y_bottom = future_pos[1] - height / 2.0
        y_top = future_pos[1] + height / 2.0
        
        vertices = np.array([
            [x_left, y_bottom],
            [x_right, y_bottom],
            [x_right, y_top],
            [x_left, y_top]
        ])
        
        return vertices

    def laserscan_callback(self, data):

        timestamp = data.header.stamp.to_sec()
        points = []
        
        for i, range_val in enumerate(data.ranges):
            if i % 1 == 0:
                if range_val >= data.range_min and range_val <= data.range_max:
                    angle = data.angle_min + i * data.angle_increment
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    points.append([x, y, timestamp])
        
        with self.points_lock:
            self.points = np.array(points)
            self.timestamp = timestamp
        
        

    def merge_clusters(self, clusters):

        merged_clusters = []
        used = set()
        
        for i, cluster in enumerate(clusters):
            if i in used:
                continue
            merged_points = cluster['points']
            merged_centroid = cluster['centroid']
            merged_label = cluster['label']
            merged_ids = {cluster.get('id', -1)}
            merged_velocity = self.cluster_velocity_history.get(cluster.get('id', -1), [np.array([0, 0])])[-1]
            
            for j, other_cluster in enumerate(clusters[i+1:], start=i+1):
                if j in used:
                    continue
                dist = np.linalg.norm(cluster['centroid'] - other_cluster['centroid'])
                other_velocity = self.cluster_velocity_history.get(other_cluster.get('id', -1), [np.array([0, 0])])[-1]
                velocity_diff = np.linalg.norm(merged_velocity - other_velocity)

                if dist < self.eps * 1.5 or (velocity_diff < self.velocity_threshold * 2 and dist < self.eps * 2):
                    merged_points = np.vstack((merged_points, other_cluster['points']))
                    merged_centroid = np.mean(merged_points[:, :2], axis=0)
                    merged_ids.add(other_cluster.get('id', -1))
                    used.add(j)
            
            merged_clusters.append({
                'points': merged_points,
                'centroid': merged_centroid,
                'label': merged_label,
                'id': min(merged_ids) if merged_ids else self.next_cluster_id
            })
            used.add(i)
        
        return merged_clusters

    def cluster_points(self):

        with self.points_lock:
            if self.points is None or len(self.points) == 0:
                return [], None, None
            points = self.points.copy()
        
        ranges = np.linalg.norm(points[:, :2], axis=1)
        avg_range = np.mean(ranges) if len(ranges) > 0 else 1.0
        dynamic_eps = min(0.5, max(0.2, self.eps * (avg_range / 2.0)))
        dynamic_min_samples = max(5, int(len(points) * 0.02))
        
        clustering = DBSCAN(eps=dynamic_eps, min_samples=dynamic_min_samples).fit(points[:, :2])
        labels = clustering.labels_
        
        if len(labels) != len(points):
            rospy.logwarn(f"Không khớp: labels có {len(labels)} phần tử, points có {len(points)} phần tử")
            return [], None, None
        
        clusters = []
        for label in set(labels) - {-1}:
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points[:, :2], axis=0)
            clusters.append({
                'points': cluster_points,
                'centroid': centroid,
                'label': label
            })
        
        if clusters:
            clusters = self.merge_clusters(clusters)
        

        return clusters, points, labels

    def track_dynamic_objects(self, clusters, dt):

        dynamic_objects = []
        unmatched_filters = set(self.kalman_filters.keys())
        matched_ids = set()
        deleted_ids = []
        
        for cluster in clusters:
            min_dist = float('inf')
            matched_id = None
            matched_kf = None
            matched_prev_cluster = None
            

            for kf_id, kf in self.kalman_filters.items():
                if kf_id in matched_ids:
                    continue
                kf.predict() 
                predicted_pos = kf.x[:2]
                dist = np.linalg.norm(cluster['centroid'] - predicted_pos)
                dynamic_match_distance = self.max_match_distance + np.linalg.norm(kf.x[2:4]) * dt  
                if dist < min_dist and dist < dynamic_match_distance:
                    min_dist = dist
                    matched_id = kf_id
                    matched_kf = kf
            

            for prev_cluster in self.prev_clusters:
                dist = np.linalg.norm(cluster['centroid'] - prev_cluster['centroid'])
                if dist < min_dist and dist < self.max_match_distance:
                    min_dist = dist
                    matched_prev_cluster = prev_cluster
            

            if matched_id is None:
                cluster['id'] = self.next_cluster_id
                cluster['motion_history'] = []
                self.kalman_filters[self.next_cluster_id] = self.initialize_kalman_filter(cluster['centroid'], dt)
                self.cluster_age[self.next_cluster_id] = 0
                self.cluster_point_counts[self.next_cluster_id] = [len(cluster['points'])]
                self.cluster_velocity_history[self.next_cluster_id] = [np.array([0.0, 0.0])]
                cluster['future_position'] = cluster['centroid']
                cluster['velocity'] = np.array([0.0, 0.0])
                self.next_cluster_id += 1
            else:
                if matched_id in unmatched_filters:
                    unmatched_filters.remove(matched_id)
                    matched_ids.add(matched_id)
                else:
                    rospy.logwarn(f"ID {matched_id} không có trong unmatched_filters, bỏ qua cập nhật Kalman")
                    continue
                matched_kf.update(cluster['centroid'])
                self.cluster_age[matched_id] = 0
                cluster['id'] = matched_id
                cluster['velocity'] = matched_kf.x[2:4]
                cluster['motion_history'] = matched_prev_cluster.get('motion_history', []) if matched_prev_cluster else []
                cluster['future_position'] = self.predict_future_state(matched_kf, self.prediction_time, dt)
                self.cluster_point_counts[matched_id].append(len(cluster['points']))
                self.cluster_point_counts[matched_id] = self.cluster_point_counts[matched_id][-15:]
                self.cluster_velocity_history[matched_id].append(cluster['velocity'])
                self.cluster_velocity_history[matched_id] = self.cluster_velocity_history[matched_id][-15:]

            if 'future_position' in cluster:
                vertices = self.compute_convex_hull_rectangle(cluster['points'], cluster['future_position'])
                cluster['vertices'] = vertices
            else:
                rospy.logwarn(f"Cụm ID {cluster['id']} thiếu future_position, bỏ qua tính toán hình chữ nhật")
                continue

            is_dynamic = cluster['id'] in self.dynamic_object_ids

            if not is_dynamic and matched_prev_cluster:
                velocities = []
                current_points = cluster['points'][:, :2]
                prev_points = matched_prev_cluster['points'][:, :2]
                
                for curr_p in current_points:
                    distances = np.linalg.norm(prev_points - curr_p, axis=1)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < self.max_match_distance:
                        velocity = (curr_p - prev_points[min_idx]) / dt
                        if np.linalg.norm(velocity) > self.velocity_threshold:
                            velocities.append(velocity)
                
                if len(velocities) >= self.min_moving_points_ratio * len(current_points):
                    angles = [np.arctan2(v[1], v[0]) for v in velocities]
                    angle_std = np.std(angles)
                    mean_velocity = np.mean(velocities, axis=0) if velocities else np.array([0, 0])
                    
                    cluster['motion_history'].append(np.linalg.norm(mean_velocity))
                    cluster['motion_history'] = cluster['motion_history'][-15:]
                    
                    if angle_std < self.angle_std_threshold and len(cluster['motion_history']) >= 2:
                        avg_motion = np.mean(cluster['motion_history'])
                        if avg_motion > self.velocity_threshold:
                            is_dynamic = True
                            cluster['velocity'] = mean_velocity
                            self.dynamic_object_ids.add(cluster['id'])
                
                if not is_dynamic and matched_prev_cluster:
                    centroid_velocity = np.linalg.norm(cluster['centroid'] - matched_prev_cluster['centroid']) / dt
                    current_count = len(cluster['points'])
                    prev_count = len(matched_prev_cluster['points'])
                    point_count_history = self.cluster_point_counts.get(cluster['id'], [current_count])
                    
                    if len(point_count_history) >= 2:
                        avg_point_count = np.mean(point_count_history)
                        point_count_variation = abs(current_count - avg_point_count) / avg_point_count if avg_point_count > 0 else 0
                        
                        current_range = np.linalg.norm(cluster['centroid'])
                        prev_range = np.linalg.norm(matched_prev_cluster['centroid'])
                        if prev_range > 0 and current_range > 0:
                            points = cluster['points'][:, :2]
                            angles = np.arctan2(points[:, 1], points[:, 0])
                            angular_span = np.ptp(angles)
                            expected_count = prev_count * (prev_range / current_range) * (angular_span / self.angular_resolution)
                            expected_count = max(1, expected_count)
                            adjusted_variation = abs(current_count - expected_count) / expected_count if expected_count > 0 else 0
                        else:
                            expected_count = prev_count
                            adjusted_variation = point_count_variation
                        
                        if centroid_velocity > self.velocity_threshold and adjusted_variation < self.point_count_variation_threshold:
                            is_dynamic = True
                            cluster['velocity'] = (cluster['centroid'] - matched_prev_cluster['centroid']) / dt
                            cluster['motion_history'].append(centroid_velocity)
                            cluster['motion_history'] = cluster['motion_history'][-15:]
                            self.dynamic_object_ids.add(cluster['id'])
                            rospy.loginfo(f"Cụm ID {cluster['id']} là động: vận tốc tâm {centroid_velocity:.3f} m/s, biến thiên số điểm {adjusted_variation:.3f}")
                        elif centroid_velocity > self.static_velocity_threshold:
                            rospy.loginfo(f"Cụm ID {cluster['id']} có vận tốc tâm {centroid_velocity:.3f} m/s nhưng không được coi là tĩnh")
            
            if is_dynamic:
                dynamic_objects.append(cluster)
                rospy.loginfo(f"Vật thể động ID {cluster['id']}: Tâm ({cluster['centroid'][0]:.2f}, {cluster['centroid'][1]:.2f}), "
                              f"Vị trí tương lai ({cluster['future_position'][0]:.2f}, {cluster['future_position'][1]:.2f})")

        for kf_id in list(unmatched_filters):
            self.cluster_age[kf_id] += 1
            if self.cluster_age[kf_id] > self.max_age:
                if kf_id in self.dynamic_object_ids:
                    deleted_ids.append(kf_id)
                    rospy.loginfo(f"vat can can xoa {kf_id} vật cản động bị xóa")
                del self.kalman_filters[kf_id]
                del self.cluster_age[kf_id]
                del self.cluster_point_counts[kf_id]
                del self.cluster_velocity_history[kf_id]
                self.dynamic_object_ids.discard(kf_id)
        
        self.prev_clusters = clusters
        self.publish_dynamic_obstacles(dynamic_objects, deleted_ids)
        
        return dynamic_objects

    def publish_dynamic_obstacles(self, dynamic_objects, deleted_ids):

        multi_array = Float64MultiArray()
        multi_array.layout.dim = []
        multi_array.layout.data_offset = 0

        data = []
        for deleted_id in deleted_ids:
            data.append(float(deleted_id))
            for _ in range(4):
                data.append(0.0)
                data.append(0.0)

        for obj in dynamic_objects:
            data.append(float(obj['id']))
            for vertex in obj['vertices']:
                data.append(self.L - vertex[1])
                data.append(vertex[0])
        
        multi_array.data = data

        num_publishes = 2  
        for i in range(num_publishes):
            self.dynamic_obstacles_pub.publish(multi_array)
            rospy.loginfo(f"[{i+1}/{num_publishes}] Đã xuất bản {len(dynamic_objects)} vật cản động và {len(deleted_ids)} vật cản động bị xóa: {deleted_ids}")
            rospy.sleep(0.02)  
        

        if not data:
            rospy.logwarn("Không có dữ liệu vật cản động hoặc xóa để xuất bản")

    def visualize(self):

        with self.points_lock:
            if self.timestamp is None or self.timestamp == self.last_processed_timestamp:
                return
            timestamp = self.timestamp
            last_processed_timestamp = self.timestamp
            
            self.dt_history = []

            dt = timestamp - self.last_processed_timestamp if self.last_processed_timestamp else 0.14
            self.dt_history.append(dt)
            self.dt_history = self.dt_history[-10:]  
            dt = np.mean(self.dt_history)  
        
        if dt < 0.01:
            rospy.logwarn(f"dt không hợp lệ: {dt}, bỏ qua khung hình")
            return
        
        clusters, points, labels = self.cluster_points()
        if points is None or len(points) == 0:
            rospy.loginfo("Không có điểm hợp lệ trong lần quét này, bỏ qua")
            return
        
        dynamic_objects = self.track_dynamic_objects(clusters, dt)
        
        plt.clf()
        ax = plt.gca()
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)

        ax.set_xticks(np.arange(0, 6, 0.5)) 
        ax.set_yticks(np.arange(0, 5, 0.5))  
        
        noise_points = points[labels == -1]
        if len(noise_points) > 0:
            ax.scatter(noise_points[:, 0], noise_points[:, 1], c='lightgray', s=5, alpha=0.3, label='Nhiễu')
        
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters), 1)))
        for i, cluster in enumerate(clusters):
            cluster_points = cluster['points']
            centroid = cluster['centroid']
            is_dynamic = any(obj['id'] == cluster['id'] for obj in dynamic_objects)
            if not is_dynamic:
                color = colors[i % len(colors)]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=8, alpha=0.5, label=f'Cụm tĩnh {cluster["id"]}')
                ax.plot(centroid[0], centroid[1], 'o', color=color, markersize=6, alpha=0.5)
                ax.text(centroid[0], centroid[1], f'ID {cluster["id"]}', fontsize=8, color='black', alpha=0.5)
        
        for obj in dynamic_objects:
            cluster_points = obj['points']
            centroid = obj['centroid']
            velocity = obj['velocity']
            future_pos = obj['future_position']
            vertices = obj.get('vertices', np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c='red', s=20, label=f'Vật thể động {obj["id"]}', zorder=10)
            ax.plot(centroid[0], centroid[1], 'o', color='darkred', markersize=12, zorder=11)
            ax.text(centroid[0] + 0.05, centroid[1] + 0.05, f'ID {obj["id"]}', fontsize=10, color='black', weight='bold', zorder=12)
            ax.arrow(centroid[0], centroid[1], velocity[0] * 0.3, velocity[1] * 0.3, 
                    head_width=0.08, head_length=0.1, color='blue', linewidth=2, zorder=13)
            ax.text(centroid[0] + 0.15, centroid[1] + 0.15, 
                    f"{np.linalg.norm(velocity):.2f} m/s", fontsize=10, color='blue', weight='bold', zorder=14)
            
            ax.plot(future_pos[0], future_pos[1], 'x', color='green', markersize=15, label=f'Tương lai ID {obj["id"]}', zorder=12)
            ax.text(future_pos[0] + 0.05, future_pos[1] + 0.05, 
                    f'Tương lai ID {obj["id"]}', fontsize=10, color='green', weight='bold', zorder=13)
            
            vertices_closed = np.vstack((vertices, vertices[0]))
            polygon = Polygon(vertices_closed, closed=True, edgecolor='purple', facecolor='none', linestyle='--', linewidth=2, zorder=11)
            ax.add_patch(polygon)
            ax.text(vertices[0, 0], vertices[0, 1] + 0.05, f'Hull ID {obj["id"]}', fontsize=10, color='purple', weight='bold', zorder=12)
        
        plt.title(f"RPLIDAR A1M8: {len(dynamic_objects)} Vật thể động, Dự đoán {self.prediction_time:.1f}s", fontsize=12, weight='bold')
        plt.xlabel("X (m)", fontsize=10)
        plt.ylabel("Y (m)", fontsize=10)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        
        plt.draw()
        plt.pause(0.01)
        self.last_processed_timestamp = last_processed_timestamp

    def run(self):

        while not rospy.is_shutdown():
            self.visualize()
            self.rate.sleep()
            if self.cluster_age:
                deleted_ids = [kf_id for kf_id in self.cluster_age if self.cluster_age[kf_id] > self.max_age]
                if deleted_ids:
                    self.publish_dynamic_obstacles([], deleted_ids) 
                    for kf_id in deleted_ids:
                        if kf_id in self.dynamic_object_ids:
                            rospy.loginfo(f"Ưu tiên xóa vật cản động ID {kf_id}")
                        del self.kalman_filters[kf_id]
                        del self.cluster_age[kf_id]
                        del self.cluster_point_counts[kf_id]
                        del self.cluster_velocity_history[kf_id]
                        self.dynamic_object_ids.discard(kf_id)

if __name__ == '__main__':
    try:
        processor = LaserScanProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node đã dừng.")