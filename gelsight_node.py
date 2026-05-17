#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge

from config import GSConfig
from utilities.reconstruction import Reconstruction3D
from utilities.gelsightmini import GelSightMini


class GelSightHolePatternNode(Node):
    def __init__(self):
        super().__init__("gelsight_hole_pattern_node")

        self.declare_parameter("gs_config_path", "default_config.json")
        self.declare_parameter("camera_index", 1)

        self.declare_parameter("raw_image_topic", "/gelsight/image_raw")
        self.declare_parameter("masked_image_topic", "/gelsight/hole_pattern_masked")
        self.declare_parameter("pose_topic", "/gelsight/hole_pattern_pose")

        self.declare_parameter("depth_threshold", 0.5)
        self.declare_parameter("min_blob_area", 30.0)
        self.declare_parameter("max_blob_area", 50000.0)
        self.declare_parameter("min_contact_area", 10000.0)

        # Black marker detection parameters
        self.declare_parameter("marker_threshold", 100)
        self.declare_parameter("marker_min_area", 1.0)
        self.declare_parameter("marker_max_area", 8000.0)
        self.declare_parameter("marker_min_circularity", 0.010)
        self.declare_parameter("use_adaptive_marker_threshold", True)

        self.gs_config_path = self.get_parameter("gs_config_path").value
        self.camera_index = self.get_parameter("camera_index").value

        self.raw_image_topic = self.get_parameter("raw_image_topic").value
        self.masked_image_topic = self.get_parameter("masked_image_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value

        self.depth_threshold = self.get_parameter("depth_threshold").value
        self.min_blob_area = self.get_parameter("min_blob_area").value
        self.max_blob_area = self.get_parameter("max_blob_area").value
        self.min_contact_area = self.get_parameter("min_contact_area").value

        self.marker_threshold = int(self.get_parameter("marker_threshold").value)
        self.marker_min_area = self.get_parameter("marker_min_area").value
        self.marker_max_area = self.get_parameter("marker_max_area").value
        self.marker_min_circularity = self.get_parameter("marker_min_circularity").value
        self.use_adaptive_marker_threshold = self.get_parameter(
            "use_adaptive_marker_threshold"
        ).value

        self.bridge = CvBridge()

        self.gs_config = GSConfig(self.gs_config_path)
        self.config = self.gs_config.config

        self.reconstruction = Reconstruction3D(
            image_width=self.config.camera_width,
            image_height=self.config.camera_height,
            use_gpu=self.config.use_gpu,
        )

        if self.reconstruction.load_nn(self.config.nn_model_path) is None:
            raise RuntimeError("Failed to load GelSight reconstruction model")

        self.cam_stream = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height,
        )

        devices = self.cam_stream.get_device_list()
        self.get_logger().info(f"Available camera devices: {devices}")
        self.get_logger().info(f"self.camera_index: {self.camera_index}")

        self.cam_stream.select_device(self.camera_index)
        self.cam_stream.start()

        self.raw_pub = self.create_publisher(Image, self.raw_image_topic, 10)
        self.masked_pub = self.create_publisher(Image, self.masked_image_topic, 10)
        self.pose_pub = self.create_publisher(Pose2D, self.pose_topic, 10)

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info("GelSight hole pattern node started")
        self.get_logger().info(f"Raw image topic: {self.raw_image_topic}")
        self.get_logger().info(f"Masked image topic: {self.masked_image_topic}")
        self.get_logger().info(f"Pose topic: {self.pose_topic}")

    def timer_callback(self):
        frame = self.cam_stream.update(dt=0)

        if frame is None:
            return

        # GelSightMini returns BGR frame.
        frame_bgr = frame.copy()

        # Resize if the camera opens at full resolution.
        frame_bgr = cv2.resize(
            frame_bgr,
            (self.config.camera_width, self.config.camera_height),
            interpolation=cv2.INTER_AREA,
        )

        raw_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = "gelsight_camera"
        self.raw_pub.publish(raw_msg)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        output = self.process_frame(frame_rgb)

        if output is None:
            return

        debug_bgr, pose = output

        masked_msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8")
        masked_msg.header.stamp = raw_msg.header.stamp
        masked_msg.header.frame_id = "gelsight_camera"
        self.masked_pub.publish(masked_msg)

        if pose is not None:
            pose_msg = Pose2D()
            pose_msg.x = float(pose["cx"])
            pose_msg.y = float(pose["cy"])
            pose_msg.theta = float(pose["theta"])
            self.pose_pub.publish(pose_msg)

    def process_frame(self, image_rgb):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------
        # 1. Black marker / dot detection from raw GelSight image
        # -------------------------------------------------------
        marker_blobs, marker_mask = self.detect_black_marker_blobs(image_bgr)

        # -------------------------------------------------------
        # 2. Depth reconstruction for contact/hole pose
        # -------------------------------------------------------
        depth_map, contact_mask, grad_x, grad_y = self.reconstruction.get_depthmap(
            image=image_rgb,
            markers_threshold=(
                self.config.marker_mask_min,
                self.config.marker_mask_max,
            ),
        )

        if depth_map is None:
            return None

        if np.isnan(depth_map).any():
            self.get_logger().warn("Depth map contains NaNs. Skipping frame.")
            return None

        # Threshold depth
        threshold_mask = (depth_map >= self.depth_threshold).astype(np.uint8) * 255

        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            threshold_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            debug_bgr = self.make_debug_image(
                image_rgb=image_rgb,
                contact_mask=threshold_mask,
                marker_mask=marker_mask,
                circle=None,
                pose=None,
                marker_blobs=marker_blobs,
            )
            return debug_bgr, None

        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if self.min_blob_area <= area <= self.max_blob_area:
                valid_contours.append(cnt)

        if len(valid_contours) == 0:
            debug_bgr = self.make_debug_image(
                image_rgb=image_rgb,
                contact_mask=threshold_mask,
                marker_mask=marker_mask,
                circle=None,
                pose=None,
                marker_blobs=marker_blobs,
            )
            return debug_bgr, None

        main_contour = max(valid_contours, key=cv2.contourArea)
        contact_area = cv2.contourArea(main_contour)

        if contact_area < self.min_contact_area:
            debug_bgr = self.make_debug_image(
                image_rgb=image_rgb,
                contact_mask=threshold_mask,
                marker_mask=marker_mask,
                circle=None,
                pose=None,
                marker_blobs=marker_blobs,
            )
            return debug_bgr, None

        main_mask = np.zeros_like(threshold_mask)
        cv2.drawContours(main_mask, [main_contour], -1, 255, thickness=-1)

        circle = self.fit_circle(main_contour)
        pose = self.compute_pca_pose(main_mask)

        debug_bgr = self.make_debug_image(
            image_rgb=image_rgb,
            contact_mask=main_mask,
            marker_mask=marker_mask,
            circle=circle,
            pose=pose,
            marker_blobs=marker_blobs,
        )

        return debug_bgr, pose

    def detect_black_marker_blobs(self, image_bgr):
        """
        Detect black circular marker dots from the raw GelSight image.
        This is separate from depth/contact detection.
        """

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if self.use_adaptive_marker_threshold:
            marker_mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                5,
            )
        else:
            _, marker_mask = cv2.threshold(
                gray,
                self.marker_threshold,
                255,
                cv2.THRESH_BINARY_INV,
            )

        kernel = np.ones((3, 3), np.uint8)
        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        params.filterByArea = True
        params.minArea = float(self.marker_min_area)
        params.maxArea = float(self.marker_max_area)

        params.filterByCircularity = True
        params.minCircularity = float(self.marker_min_circularity)

        params.filterByConvexity = False

        params.filterByInertia = True
        params.minInertiaRatio = 0.010

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(marker_mask)

        blobs = []

        for kp in keypoints:
            blobs.append(
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "radius": float(kp.size / 2.0),
                }
            )

        return blobs, marker_mask

    def fit_circle(self, contour):
        if contour is None or len(contour) < 5:
            return None

        (x, y), radius = cv2.minEnclosingCircle(contour)

        return {
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
        }

    def compute_pca_pose(self, binary_mask):
        ys, xs = np.where(binary_mask > 0)

        if len(xs) < 5:
            return None

        points = np.column_stack((xs, ys)).astype(np.float32)

        mean = np.mean(points, axis=0)
        centered = points - mean

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        principal_axis = eigenvectors[:, 0]
        theta = np.arctan2(principal_axis[1], principal_axis[0])

        return {
            "cx": float(mean[0]),
            "cy": float(mean[1]),
            "theta": float(theta),
            "axis_x": float(principal_axis[0]),
            "axis_y": float(principal_axis[1]),
            "lambda_1": float(eigenvalues[0]),
            "lambda_2": float(eigenvalues[1]),
        }

    def make_debug_image(
        self,
        image_rgb,
        contact_mask,
        marker_mask,
        circle,
        pose,
        marker_blobs,
    ):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        debug = image_bgr.copy()

        # Draw contact/depth contour in green
        contours, _ = cv2.findContours(
            contact_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)

        # Draw black marker detections in yellow
        for blob in marker_blobs:
            x = int(blob["x"])
            y = int(blob["y"])
            r = max(2, int(blob["radius"]))

            cv2.circle(debug, (x, y), r, (0, 255, 255), 2)
            cv2.circle(debug, (x, y), 2, (0, 0, 255), -1)

        # Draw contact/hole enclosing circle in magenta
        if circle is not None:
            cx = int(circle["x"])
            cy = int(circle["y"])
            radius = int(circle["radius"])

            cv2.circle(debug, (cx, cy), radius, (255, 0, 255), 2)
            cv2.circle(debug, (cx, cy), 3, (0, 0, 255), -1)

        # Draw PCA axis in red
        if pose is not None:
            cx = int(pose["cx"])
            cy = int(pose["cy"])
            theta = pose["theta"]

            axis_len = 70

            x1 = int(cx - axis_len * np.cos(theta))
            y1 = int(cy - axis_len * np.sin(theta))
            x2 = int(cx + axis_len * np.cos(theta))
            y2 = int(cy + axis_len * np.sin(theta))

            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(debug, (cx, cy), 5, (255, 255, 255), -1)

            text = (
                f"pose x={pose['cx']:.1f}, y={pose['cy']:.1f}, "
                f"theta={pose['theta']:.2f}, markers={len(marker_blobs)}"
            )
        else:
            text = f"markers={len(marker_blobs)}"

        cv2.putText(
            debug,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        contact_mask_bgr = cv2.cvtColor(contact_mask, cv2.COLOR_GRAY2BGR)
        marker_mask_bgr = cv2.cvtColor(marker_mask, cv2.COLOR_GRAY2BGR)

        masked_image = cv2.bitwise_and(image_bgr, contact_mask_bgr)

        stacked = np.hstack(
            [
                image_bgr,
                contact_mask_bgr,
                marker_mask_bgr,
                masked_image,
                debug,
            ]
        )

        return stacked

    def destroy_node(self):
        try:
            if self.cam_stream.camera is not None:
                self.cam_stream.camera.release()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = GelSightHolePatternNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()