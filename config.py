"""
Configuration module for Unattended Object Detection System
Handles all system settings, thresholds, and parameters
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoConfig:
    """Video processing configuration"""
    supported_formats: List[str] = field(default_factory=lambda: [
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp'
    ])
    process_every_n_frames: int = 1  # Process every N frames (1 = every frame)
    buffer_fps: float = 15.0  # FPS for saved event clips
    pre_buffer_seconds: float = 5.0  # Seconds before event to include in clip
    post_buffer_seconds: float = 5.0  # Seconds after event to include in clip


@dataclass
class DetectionConfig:
    """YOLO detection configuration"""
    model_size: str = "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    img_size: int = 640  # Input image size for YOLO
    conf_threshold: float = 0.35  # Confidence threshold for detections
    iou_threshold: float = 0.45  # IoU threshold for NMS
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    verbose: bool = False  # YOLO verbose output

    # Object classes to monitor (COCO class names)
    object_classes: List[str] = field(default_factory=lambda: [
        "backpack", "handbag", "suitcase", "laptop", "bag", 
        "cell phone", "bottle", "box", "sports ball", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush", "umbrella", "tie", "remote", "keyboard",
        "mouse", "microwave", "oven", "toaster", "sink",
        "refrigerator", "blender", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake"
    ])

    person_class_name: str = "person"


@dataclass
class TrackingConfig:
    """DeepSORT tracking configuration"""
    max_age: int = 30  # Maximum frames to keep track without detection
    n_init: int = 3  # Minimum consecutive detections for track initialization
    max_iou_distance: float = 0.7  # Maximum IoU distance for association
    max_cosine_distance: float = 0.2  # Maximum cosine distance for appearance
    nn_budget: int = 100  # Maximum size of appearance descriptor gallery


@dataclass
class AlertConfig:
    """Alert and abandonment detection configuration"""
    t_unattended: float = 3.0  # Seconds before object considered unattended
    confirm_frames: int = 3  # Consecutive frames to confirm condition
    motion_threshold_px: float = 2.5  # Pixel movement threshold per frame
    proximity_threshold_px: float = 120.0  # Person-object proximity in pixels
    contact_window_seconds: float = 2.0  # Time window for person-object contact
    stationary_buffer_size: int = 16  # Number of frames for motion calculation

    # Alert cooldown to prevent spam
    alert_cooldown_seconds: float = 10.0  # Minimum time between alerts for same object


@dataclass
class SystemConfig:
    """System-wide configuration"""
    project_root: str = os.getcwd()
    video_path: str = "video.mp4"
    events_dir: str = "events"
    logs_dir: str = "logs"
    models_dir: str = "models"

    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_to_console: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # Performance settings
    enable_gpu: bool = True
    num_threads: int = -1  # -1 for auto-detect
    memory_limit_gb: Optional[float] = None  # None for no limit


class ConfigManager:
    """Manages application configuration"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.video = VideoConfig()
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.alert = AlertConfig()
        self.system = SystemConfig()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            if 'video' in config_data:
                self._update_config(self.video, config_data['video'])
            if 'detection' in config_data:
                self._update_config(self.detection, config_data['detection'])
            if 'tracking' in config_data:
                self._update_config(self.tracking, config_data['tracking'])
            if 'alert' in config_data:
                self._update_config(self.alert, config_data['alert'])
            if 'system' in config_data:
                self._update_config(self.system, config_data['system'])

            print(f"✓ Configuration loaded from {config_file}")
        except Exception as e:
            print(f"⚠ Failed to load config from {config_file}: {e}")

    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML file"""
        try:
            config_data = {
                'video': self._config_to_dict(self.video),
                'detection': self._config_to_dict(self.detection),
                'tracking': self._config_to_dict(self.tracking),
                'alert': self._config_to_dict(self.alert),
                'system': self._config_to_dict(self.system)
            }

            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            print(f"✓ Configuration saved to {config_file}")
        except Exception as e:
            print(f"⚠ Failed to save config to {config_file}: {e}")

    def _update_config(self, config_obj: object, data: Dict[str, Any]) -> None:
        """Update configuration object with dictionary data"""
        for key, value in data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def _config_to_dict(self, config_obj: object) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return {k: v for k, v in config_obj.__dict__.items() if not k.startswith('_')}

    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute based on project root"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.system.project_root, relative_path)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        dirs_to_create = [
            self.get_absolute_path(self.system.events_dir),
            self.get_absolute_path(self.system.logs_dir),
            self.get_absolute_path(self.system.models_dir)
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

    def validate_video_format(self, video_path: str) -> bool:
        """Check if video format is supported"""
        file_ext = Path(video_path).suffix.lower()
        return file_ext in self.video.supported_formats

    def get_model_path(self) -> str:
        """Get path for YOLO model file"""
        model_file = self.detection.model_size
        if not model_file.endswith('.pt'):
            model_file += '.pt'

        # Check if model exists in models directory
        local_model_path = self.get_absolute_path(
            os.path.join(self.system.models_dir, model_file)
        )

        if os.path.exists(local_model_path):
            return local_model_path

        # Return model name for ultralytics to download
        return self.detection.model_size

    def print_summary(self) -> None:
        """Print configuration summary"""
        print("\n" + "="*50)
        print("🔧 CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Video Format Support: {', '.join(self.video.supported_formats)}")
        print(f"YOLO Model: {self.detection.model_size}")
        print(f"Detection Confidence: {self.detection.conf_threshold}")
        print(f"Unattended Threshold: {self.alert.t_unattended}s")
        print(f"Motion Threshold: {self.alert.motion_threshold_px}px")
        print(f"Proximity Threshold: {self.alert.proximity_threshold_px}px")
        print(f"GPU Enabled: {self.system.enable_gpu}")
        print(f"Log Level: {self.system.log_level}")
        print("="*50)


# Default configuration instance
default_config = ConfigManager()

# Example usage and configuration validation
if __name__ == "__main__":
    config = ConfigManager()
    config.print_summary()

    # Example: Save default configuration
    config.save_to_file("config.yaml")
