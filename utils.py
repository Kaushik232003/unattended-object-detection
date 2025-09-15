"""
Utility functions for Unattended Object Detection System
Contains helper functions for detection, tracking, logging, and video processing
"""

import os
import cv2
import time
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import deque
from pathlib import Path
import psutil
import torch


class Logger:
    """Enhanced logging utility with file and console output"""

    def __init__(self, name: str, config: Any):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.system.log_level))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if config.system.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if config.system.log_to_file:
            log_dir = config.get_absolute_path(config.system.logs_dir)
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, f"{name}.log")
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.system.max_log_file_size,
                backupCount=config.system.log_backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)


class PerformanceMonitor:
    """Monitor system performance and resource usage"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.start_time = time.time()
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)

    def start_frame(self) -> float:
        """Start timing a frame processing"""
        return time.time()

    def end_frame(self, start_time: float) -> float:
        """End timing a frame and update statistics"""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        return processing_time

    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.processing_times) > 0:
            return 1.0 / np.mean(self.processing_times)
        return 0.0

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': self._get_gpu_memory() if torch.cuda.is_available() else None
        }

    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9
            }
        return {}

    def log_performance(self, interval_seconds: int = 30) -> None:
        """Log performance statistics at intervals"""
        current_time = time.time()
        if hasattr(self, 'last_log_time'):
            if current_time - self.last_log_time < interval_seconds:
                return

        stats = self.get_system_stats()
        fps = self.get_fps()

        self.logger.info(
            f"Performance: {fps:.1f} FPS, "
            f"CPU: {stats['cpu_percent']:.1f}%, "
            f"Memory: {stats['memory_percent']:.1f}%"
        )

        if stats['gpu_available'] and stats['gpu_memory']:
            self.logger.info(
                f"GPU Memory: {stats['gpu_memory']['allocated_gb']:.2f}GB allocated, "
                f"{stats['gpu_memory']['reserved_gb']:.2f}GB reserved"
            )

        self.last_log_time = current_time


class GeometryUtils:
    """Geometric calculations for bounding boxes and tracking"""

    @staticmethod
    def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def bbox_iou(bbox1: Tuple[float, float, float, float], 
                 bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = GeometryUtils.bbox_area(bbox1)
        area2 = GeometryUtils.bbox_area(bbox2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def is_point_in_bbox(point: Tuple[float, float], 
                        bbox: Tuple[float, float, float, float]) -> bool:
        """Check if point is inside bounding box"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class MotionAnalyzer:
    """Analyze motion patterns for object tracking"""

    def __init__(self, buffer_size: int = 16, motion_threshold: float = 2.5):
        self.buffer_size = buffer_size
        self.motion_threshold = motion_threshold
        self.position_history = deque(maxlen=buffer_size)

    def add_position(self, position: Tuple[float, float], timestamp: float) -> None:
        """Add new position to motion history"""
        self.position_history.append((position[0], position[1], timestamp))

    def calculate_motion_energy(self) -> float:
        """Calculate average motion energy over recent positions"""
        if len(self.position_history) < 2:
            return 0.0

        displacements = []
        history = list(self.position_history)

        for i in range(1, len(history)):
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            dt = history[i][2] - history[i-1][2]

            if dt > 0:
                displacement = np.sqrt(dx*dx + dy*dy) / dt
                displacements.append(displacement)

        return np.mean(displacements) if displacements else 0.0

    def is_stationary(self) -> bool:
        """Check if object is stationary based on motion threshold"""
        return self.calculate_motion_energy() < self.motion_threshold

    def get_velocity_vector(self) -> Tuple[float, float]:
        """Get current velocity vector"""
        if len(self.position_history) < 2:
            return (0.0, 0.0)

        latest = self.position_history[-1]
        previous = self.position_history[-2]

        dt = latest[2] - previous[2]
        if dt <= 0:
            return (0.0, 0.0)

        vx = (latest[0] - previous[0]) / dt
        vy = (latest[1] - previous[1]) / dt

        return (vx, vy)


class VideoProcessor:
    """Handle video input/output operations"""

    @staticmethod
    def open_video(video_path: str, logger: Logger) -> Optional[cv2.VideoCapture]:
        """Safely open video file with error handling"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video opened: {video_path}")
            logger.info(f"Properties: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")

            return cap

        except Exception as e:
            logger.error(f"Error opening video {video_path}: {e}")
            return None

    @staticmethod
    def save_clip(frames: List[np.ndarray], output_path: str, 
                  fps: float, logger: Logger) -> bool:
        """Save list of frames as video clip"""
        try:
            if not frames:
                logger.warning("No frames to save")
                return False

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return False

            # Write frames
            for frame in frames:
                writer.write(frame)

            writer.release()
            logger.info(f"Clip saved: {output_path} ({len(frames)} frames)")
            return True

        except Exception as e:
            logger.error(f"Error saving clip {output_path}: {e}")
            return False

    @staticmethod
    def validate_format(video_path: str, supported_formats: List[str]) -> bool:
        """Validate if video format is supported"""
        file_ext = Path(video_path).suffix.lower()
        return file_ext in supported_formats


class AlertManager:
    """Manage alert generation and cooldown periods"""

    def __init__(self, cooldown_seconds: float, logger: Logger):
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger
        self.last_alerts = {}  # track_id -> timestamp

    def can_alert(self, track_id: int) -> bool:
        """Check if enough time has passed since last alert for this track"""
        current_time = time.time()
        last_alert_time = self.last_alerts.get(track_id, 0)
        return current_time - last_alert_time >= self.cooldown_seconds

    def trigger_alert(self, track_id: int, class_name: str, bbox: Tuple[float, float, float, float], 
                     reason: str, video_name: str, t_unattended: float) -> Dict[str, Any]:
        """Trigger an alert and record timestamp"""
        current_time = time.time()
        self.last_alerts[track_id] = current_time

        alert_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'video': os.path.basename(video_name),
            'track_id': track_id,
            'class': class_name,
            'bbox': bbox,
            'reason': reason,
            't_unattended': t_unattended
        }

        # Log the alert
        alert_msg = (
            f"[ALERT] {alert_data['timestamp']} "
            f"Video={alert_data['video']} "
            f"object_track={track_id} "
            f"class={class_name} "
            f"reason={reason} "
            f"t_unattended={t_unattended}s"
        )

        print(alert_msg)  # Console output
        self.logger.warning(alert_msg)  # Log file

        return alert_data

    def reset_alert(self, track_id: int) -> None:
        """Reset alert status for a track (e.g., when object is picked up)"""
        if track_id in self.last_alerts:
            del self.last_alerts[track_id]


class DeviceManager:
    """Manage GPU/CPU device selection and optimization"""

    @staticmethod
    def get_optimal_device(config: Any, logger: Logger) -> str:
        """Determine the best device to use for inference"""
        if config.detection.device != "auto":
            device = config.detection.device
            logger.info(f"Using specified device: {device}")
            return device

        # Auto-detection logic
        if torch.cuda.is_available() and config.system.enable_gpu:
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and config.system.enable_gpu:
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")

        return device

    @staticmethod
    def optimize_torch_settings(device: str, logger: Logger) -> None:
        """Optimize PyTorch settings for the selected device"""
        try:
            if device == "cuda":
                # Enable optimizations for CUDA
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("Enabled CUDA optimizations")

            # Set number of threads for CPU inference
            if device == "cpu":
                num_threads = max(1, os.cpu_count() // 2)
                torch.set_num_threads(num_threads)
                logger.info(f"Set CPU threads: {num_threads}")

        except Exception as e:
            logger.warning(f"Failed to optimize torch settings: {e}")


# Utility functions for common operations
def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)


def generate_timestamp() -> str:
    """Generate timestamp string for file naming"""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def validate_dependencies(logger: Logger) -> bool:
    """Validate that all required dependencies are available"""
    required_modules = [
        'cv2', 'numpy', 'torch', 'ultralytics', 
        'deep_sort_realtime', 'yaml', 'psutil'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return False

    logger.info("All dependencies validated successfully")
    return True
