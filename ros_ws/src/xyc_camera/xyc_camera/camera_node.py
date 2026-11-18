import threading

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class CameraNode(Node):
    def __init__(self, **kwargs):
        super().__init__('camera_node', **kwargs)

        # 参数：不同相机共用代码，通过参数选择设备。
        self.declare_parameter('device', '')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('jpeg_quality', 75)

        device = self.get_parameter('device').value
        self.frame_id = self.get_parameter('frame_id').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        if not device or not self.frame_id:
            raise ValueError('Please set device and frame_id parameters')
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError('width, height and fps must be positive')
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in 1..100')

        # 创建图像消息转换器和发布者。
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, 'image_raw', qos_profile_sensor_data
        )
        self.compressed_publisher = self.create_publisher(
            CompressedImage, 'image_raw/compressed', qos_profile_sensor_data
        )
        self.encode_condition = threading.Condition()
        self.encode_pending = None
        self.encode_stopped = False

        # 打开相机，设置采集格式。
        self.camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f'Cannot open camera: {device}')

        self.camera.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')
        )
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.encode_worker = threading.Thread(target=self.encode_frame, daemon=True)
        self.encode_worker.start()

        # V4L2 read() 会等待下一幅图像。让读取定时器比相机帧率快，
        # 避免 30 Hz 定时器与相机自身 30 Hz 采集时钟相互错过。
        self.timer = self.create_timer(1.0 / (fps * 2.0), self.publish_frame)
        self.get_logger().info(
            f'Publishing {self.publisher.topic_name} from {device}'
        )

    def publish_frame(self):
        ok, frame = self.camera.read()
        if not ok or frame is None:
            if self.encode_stopped or not rclpy.ok():
                return
            raise RuntimeError('Failed to read camera frame')

        message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        if self.encode_stopped or not rclpy.ok():
            return
        self.publisher.publish(message)
        if self.compressed_publisher.get_subscription_count():
            # Acquisition JPEGs retain the capture cadence; HTTP preview throttles separately.
            with self.encode_condition:
                # Keep only the newest frame; JPEG encoding never blocks capture.
                self.encode_pending = (frame, message.header)
                self.encode_condition.notify()

    def encode_frame(self):
        while True:
            with self.encode_condition:
                self.encode_condition.wait_for(
                    lambda: self.encode_stopped or self.encode_pending is not None
                )
                if self.encode_stopped:
                    return
                frame, header = self.encode_pending
                self.encode_pending = None
            try:
                ok, encoded = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                preview = CompressedImage()
                preview.header, preview.format = header, 'jpeg'
                preview.data = encoded.tobytes()
                if not self.encode_stopped and rclpy.ok():
                    self.compressed_publisher.publish(preview)
            except Exception as exc:
                if not self.encode_stopped:
                    self.get_logger().warning(f'Preview skipped: {exc}', throttle_duration_sec=2.0)

    def destroy_node(self):
        with self.encode_condition:
            self.encode_stopped = True
            self.encode_condition.notify_all()
        self.encode_worker.join(timeout=2.0)
        self.camera.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
