#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge

from config import GSConfig, ConfigModel
from utilities.reconstruction import Reconstruction3D
from utilities.gelsightmini import GelSightMini


class GelSightPosePublisher(Node):
    """
    Estimates cable pose from GelSight depth image using PCA,
    following the method described in:

      'Cable Manipulation with a Tactile-Reactive Gripper'
       She et al., 2020  (Section IV-B)

    Pipeline:
      RGB image
        → depth map          (NN reconstruction)
        → contact mask       (depth threshold)
        → PCA on mask pixels
        → centroid  →  cable lateral position  y  (pixels)
        → principal axis angle  →  cable orientation  θ  (radians)
        → contact area  →  grasp quality  S  (bool)
    
    Published topics:
      gelsight/image_raw          sensor_msgs/Image       raw RGB
      gelsight/depth              sensor_msgs/Image       float32 depth
      gelsight/contact_mask       sensor_msgs/Image       binary mask
      gelsight/cable_angle        std_msgs/Float32        θ in radians
      gelsight/cable_pose         std_msgs/Float32MultiArray  [y, θ, quality]
      gelsight/principal_axis     geometry_msgs/Vector3Stamped  eigenvector v1
    """

    def __init__(self):
        super().__init__("gelsight_pose_publisher")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("gs_config",             "default_config.json")
        self.declare_parameter("depth_threshold",       0.1)
        self.declare_parameter("min_contact_area",      50)      # px² — below this = poor grasp
        self.declare_parameter("frame_id",              "gelsight_frame")
        self.declare_parameter("fps",                   30.0)

        config_path          = self.get_parameter("gs_config").value
        self.threshold       = self.get_parameter("depth_threshold").value
        self.min_contact_area= self.get_parameter("min_contact_area").value
        self.frame_id        = self.get_parameter("frame_id").value
        fps                  = self.get_parameter("fps").value
        print(config_path)
        
        # ── Config & reconstruction ───────────────────────────────────────────
        gs_config = GSConfig(config_path)
        self.config = gs_config.config

        self.reconstruction = Reconstruction3D(
            image_width=self.config.camera_width,
            image_height=self.config.camera_height,
            use_gpu=self.config.use_gpu,
        )
        if self.reconstruction.load_nn(self.config.nn_model_path) is None:
            self.get_logger().error("Failed to load NN model.")
            raise RuntimeError("NN model load failed.")

        # ── Camera ────────────────────────────────────────────────────────────
        self.cam_stream = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height,
        )
        devices = self.cam_stream.get_device_list()
        self.get_logger().info(f"Available devices: {devices}")
        self.cam_stream.select_device(self.config.default_camera_index)
        self.cam_stream.start()

        # ── Publishers ────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        self.pub_raw    = self.create_publisher(Image,              "gelsight/image_raw",      10)
        self.pub_depth  = self.create_publisher(Image,              "gelsight/depth",           10)
        self.pub_mask   = self.create_publisher(Image,              "gelsight/contact_mask",    10)
        self.pub_angle  = self.create_publisher(Float32,            "gelsight/cable_angle",     10)
        self.pub_pose   = self.create_publisher(Float32MultiArray,  "gelsight/cable_pose",      10)
        self.pub_axis   = self.create_publisher(Vector3Stamped,     "gelsight/principal_axis",  10)
        # debug: annotated image showing PCA overlay
        self.pub_debug  = self.create_publisher(Image,              "gelsight/debug_pca",       10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
        self.get_logger().info("GelSight pose publisher started.")

    # ── PCA pose estimation ───────────────────────────────────────────────────

    def estimate_cable_pose(self, depth_map: np.ndarray):
        """
        Implements Section IV-B of She et al. 2020.

        Steps:
          1. Threshold depth map → binary contact mask
          2. Collect contact pixel coordinates
          3. Compute centroid  → y  (lateral offset from sensor centre)
          4. Compute covariance matrix of contact pixels
          5. Eigen-decompose → principal axis v1 → angle θ

        Returns:
            y        : float  lateral offset of cable centroid from image centre (pixels)
            theta    : float  cable orientation angle (radians, w.r.t. image X axis)
            quality  : float  1.0 = good grasp, 0.0 = poor grasp
            centroid : (cx, cy) in pixel coordinates
            v1       : (vx, vy) unit eigenvector along cable axis
            v2       : (vx, vy) unit eigenvector perpendicular to cable
            eigenvalues : (λ1, λ2)
        """

        # Step 1 — threshold → contact mask
        contact_mask = depth_map >= self.threshold

        # Step 2 — collect contact pixel (x, y) coordinates
        contact_pixels = np.argwhere(contact_mask)   # shape (N, 2): [[row, col], ...]
        contact_area   = len(contact_pixels)

        # Grasp quality S: 1 if area large enough, 0 otherwise (Eq. 2 in paper)
        quality = 1.0 if contact_area >= self.min_contact_area else 0.0

        if contact_area < 2:
            # Not enough points for PCA
            return None

        # argwhere returns [row, col] → convert to [x, y] = [col, row]
        pts = contact_pixels[:, ::-1].astype(np.float64)   # shape (N, 2): [[x, y], ...]

        # Step 3 — centroid
        centroid = pts.mean(axis=0)                         # [cx, cy]
        cx, cy   = centroid

        # Lateral offset y from sensor centre (image midpoint)
        sensor_cx = depth_map.shape[1] / 2.0
        sensor_cy = depth_map.shape[0] / 2.0
        y_offset  = cy - sensor_cy                          # positive = below centre

        # Step 4 — covariance matrix  Σ = (1/N) Σ (p - μ)(p - μ)^T
        pts_centred = pts - centroid                        # shape (N, 2)
        cov = (pts_centred.T @ pts_centred) / contact_area  # 2×2 matrix

        # Step 5 — eigen decomposition
        # eigenvalues sorted ascending by np.linalg.eigh (symmetric matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal axis = eigenvector with LARGEST eigenvalue
        # np.linalg.eigh returns ascending order → last column is v1
        v1 = eigenvectors[:, -1]    # principal axis (cable direction)
        v2 = eigenvectors[:,  0]    # minor axis     (cable width direction)
        λ1 = eigenvalues[-1]
        λ2 = eigenvalues[0]

        # θ: angle of principal axis w.r.t. image X axis (radians)
        theta = np.arctan2(v1[1], v1[0])

        return {
            "y":           y_offset,
            "theta":       theta,
            "quality":     quality,
            "centroid":    (cx, cy),
            "v1":          v1,
            "v2":          v2,
            "eigenvalues": (λ1, λ2),
            "contact_area": contact_area,
        }

    # ── Debug visualisation ───────────────────────────────────────────────────

    def draw_pca_overlay(
        self,
        image: np.ndarray,
        pose: dict,
        scale: float = 60.0,
    ) -> np.ndarray:
        """
        Draws centroid + principal axes on the image, matching Fig. 4(b)
        of the paper (red = v1 principal axis, green = v2 minor axis,
        white ellipse approximation via drawn lines).
        """
        vis = image.copy()
        if pose is None:
            return vis

        cx, cy = int(pose["centroid"][0]), int(pose["centroid"][1])
        v1, v2 = pose["v1"], pose["v2"]
        λ1, λ2 = pose["eigenvalues"]

        # Scale eigenvectors by sqrt of eigenvalue for ellipse-like display
        v1_scaled = v1 * np.sqrt(λ1) * scale / max(np.sqrt(λ1), 1e-6)
        v2_scaled = v2 * np.sqrt(λ2) * scale / max(np.sqrt(λ2), 1e-6)

        # Principal axis — red  (v1, cable direction)
        p1_end = (int(cx + v1_scaled[0]), int(cy + v1_scaled[1]))
        p1_start = (int(cx - v1_scaled[0]), int(cy - v1_scaled[1]))
        cv2.arrowedLine(vis, p1_start, p1_end, (255, 0, 0), 2, tipLength=0.2)

        # Minor axis — green  (v2, cable width)
        p2_end = (int(cx + v2_scaled[0]), int(cy + v2_scaled[1]))
        p2_start = (int(cx - v2_scaled[0]), int(cy - v2_scaled[1]))
        cv2.arrowedLine(vis, p2_start, p2_end, (0, 255, 0), 2, tipLength=0.2)

        # Centroid dot
        cv2.circle(vis, (cx, cy), 5, (255, 255, 0), -1)

        # Angle text
        theta_deg = np.degrees(pose["theta"])
        cv2.putText(
            vis,
            f"theta={theta_deg:.1f}deg  y={pose['y']:.1f}px  Q={pose['quality']:.0f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        return vis

    # ── Main callback ─────────────────────────────────────────────────────────

    def timer_callback(self):
        frame = self.cam_stream.update(dt=0)
        if frame is None:
            self.get_logger().warn("No frame.", throttle_duration_sec=2.0)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stamp     = self.get_clock().now().to_msg()

        # ── Raw image ─────────────────────────────────────────────────────────
        raw_msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
        raw_msg.header.stamp    = stamp
        raw_msg.header.frame_id = self.frame_id
        self.pub_raw.publish(raw_msg)

        # ── Depth map ─────────────────────────────────────────────────────────
        depth_map, contact_mask, grad_x, grad_y = self.reconstruction.get_depthmap(
            image=frame_rgb,
            markers_threshold=(self.config.marker_mask_min, self.config.marker_mask_max),
        )

        if np.isnan(depth_map).any():
            self.get_logger().warn("NaN in depth map.", throttle_duration_sec=2.0)
            return

        # Publish float32 depth
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
        depth_msg.header.stamp    = stamp
        depth_msg.header.frame_id = self.frame_id
        self.pub_depth.publish(depth_msg)

        # Publish contact mask
        mask_u8  = (contact_mask * 255).astype(np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask_u8, encoding="mono8")
        mask_msg.header.stamp    = stamp
        mask_msg.header.frame_id = self.frame_id
        self.pub_mask.publish(mask_msg)

        # ── PCA pose estimation ───────────────────────────────────────────────
        pose = self.estimate_cable_pose(depth_map)

        if pose is None:
            self.get_logger().warn("No contact detected.", throttle_duration_sec=2.0)
            return

        # cable_angle: θ in radians
        angle_msg       = Float32()
        angle_msg.data  = float(pose["theta"])
        self.pub_angle.publish(angle_msg)

        # cable_pose: [y_offset, theta, quality]  — full state for controller
        pose_msg      = Float32MultiArray()
        pose_msg.data = [
            float(pose["y"]),
            float(pose["theta"]),
            float(pose["quality"]),
        ]
        self.pub_pose.publish(pose_msg)

        # principal_axis: v1 as a Vector3 (z=0 since it's a 2D image vector)
        axis_msg                  = Vector3Stamped()
        axis_msg.header.stamp     = stamp
        axis_msg.header.frame_id  = self.frame_id
        axis_msg.vector.x         = float(pose["v1"][0])
        axis_msg.vector.y         = float(pose["v1"][1])
        axis_msg.vector.z         = 0.0
        self.pub_axis.publish(axis_msg)

        # debug PCA overlay image
        debug_img = self.draw_pca_overlay(frame_rgb, pose)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="rgb8")
        debug_msg.header.stamp    = stamp
        debug_msg.header.frame_id = self.frame_id
        self.pub_debug.publish(debug_msg)

        self.get_logger().info(
            f"y={pose['y']:.2f}px  θ={np.degrees(pose['theta']):.1f}°  "
            f"area={pose['contact_area']}px²  Q={pose['quality']:.0f}",
            throttle_duration_sec=0.5,
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info("Shutting down.")
        if self.cam_stream.camera is not None:
            self.cam_stream.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GelSightPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()