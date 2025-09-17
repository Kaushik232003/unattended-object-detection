"""
Unattended Object Detection System
Main application for real-time detection of abandoned objects using YOLOv8 and DeepSORT
"""

# Standard library imports
import os
import sys
import time
import argparse

# Third-party imports
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any

# Custom module imports
from config import ConfigManager
from utils import Logger, PerformanceMonitor, GeometryUtils, MotionAnalyzer, VideoProcessor, AlertManager, DeviceManager, validate_dependencies, generate_timestamp, format_duration

# External libraries
try:
    import onnxruntime as ort
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def yolo_postprocess(preds, conf_threshold, coco_classes):
    """
    Postprocess YOLOv8 ONNX output.
    Args:
        preds: np.ndarray, shape [1, 84, num_detections] (YOLOv8 format)
        conf_threshold: float, confidence threshold
        coco_classes: list of class names
    Returns:
        List of dicts: [{'bbox': [x1, y1, x2, y2], 'confidence': conf, 'class': class_name, 'class_id': idx}]
    """
    detections = []
    
    # YOLOv8 output is [1, 84, num_detections], need to transpose to [num_detections, 84]
    if len(preds.shape) == 3:
        preds = preds[0].T  # Remove batch dimension and transpose
    elif len(preds.shape) == 2:
        preds = preds.T     # Just transpose if already 2D
    
    for pred in preds:
        # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]
        
        # Skip if no class scores
        if class_scores.size == 0:
            continue
            
        # Find max class score and confidence
        max_score = np.max(class_scores)
        if max_score < conf_threshold:
            continue
            
        cls_id = int(np.argmax(class_scores))
        
        # Convert center format to corner format
        x1 = float(x_center - width / 2)
        y1 = float(y_center - height / 2)
        x2 = float(x_center + width / 2)
        y2 = float(y_center + height / 2)
        
        # Ensure class_id is valid
        if cls_id >= len(coco_classes):
            continue
            
        class_name = coco_classes[cls_id]
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(max_score),
            'class': class_name,
            'class_id': cls_id
        })
    return detections


class ObjectTrack:
    """Represents a single object track with its state and history"""

    def __init__(self, track_id: int, class_name: str, config: Any):
        self.track_id = track_id
        self.class_name = class_name
        self.config = config
        # Timestamps
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_moved = time.time()
        self.last_seen_with_person = None

        # Motion analysis
        self.motion_analyzer = MotionAnalyzer(
            buffer_size=config.alert.stationary_buffer_size,
            motion_threshold=config.alert.motion_threshold_px
        )

        # State tracking
        self.stationary_frames = 0
        self.confirm_frames = 0
        self.alerted = False
        self.last_bbox = None

        # History for debugging
        self.position_history = deque(maxlen=50)
        self.detection_history = deque(maxlen=10)

    def update(self, bbox: Tuple[float, float, float, float], timestamp: float) -> None:
        """Update track with new detection"""
        self.last_seen = timestamp
        self.last_bbox = bbox

        # Update position for motion analysis
        center = GeometryUtils.bbox_center(bbox)
        self.motion_analyzer.add_position(center, timestamp)
        self.position_history.append((center, timestamp))

        # Check if object is moving
        if self.motion_analyzer.is_stationary():
            self.stationary_frames += 1
        else:
            self.last_moved = timestamp
            self.stationary_frames = 0
            self.confirm_frames = 0  # Reset confirmation
            # Reset alert if object starts moving again
            if self.alerted:
                self.alerted = False

        # Store detection for analysis
        self.detection_history.append({
            'timestamp': timestamp,
            'bbox': bbox,
            'center': center,
            'motion_energy': self.motion_analyzer.calculate_motion_energy()
        })

    def associate_with_person(self, timestamp: float) -> None:
        """Mark that this object was seen with a person"""
        self.last_seen_with_person = timestamp
        # Reset confirmation counter when person is nearby
        self.confirm_frames = 0

    def is_unattended(self, current_time: float) -> bool:
        """Check if object meets unattended criteria"""
        # Must be stationary for minimum frames
        if self.stationary_frames < self.config.alert.confirm_frames:
            return False

        # Check time since last person contact
        last_contact = self.last_seen_with_person or self.first_seen
        time_unattended = current_time - last_contact

        return time_unattended >= self.config.alert.t_unattended

    def should_alert(self, current_time: float) -> bool:
        """Check if we should trigger an alert for this track"""
        if self.alerted:
            return False

        if not self.is_unattended(current_time):
            return False

        # Increment confirmation counter
        self.confirm_frames += 1

        # Only alert after consecutive confirmations
        return self.confirm_frames >= self.config.alert.confirm_frames

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about this track"""
        current_time = time.time()
        last_contact = self.last_seen_with_person or self.first_seen

        return {
            'track_id': self.track_id,
            'class': self.class_name,
            'age': current_time - self.first_seen,
            'stationary_frames': self.stationary_frames,
            'confirm_frames': self.confirm_frames,
            'time_unattended': current_time - last_contact,
            'motion_energy': self.motion_analyzer.calculate_motion_energy(),
            'is_stationary': self.motion_analyzer.is_stationary(),
            'alerted': self.alerted,
            'last_bbox': self.last_bbox
        }


class TrackManager:
    """Manages all object tracks and their associations"""

    def __init__(self, config: Any, logger: Logger):
        self.config = config
        self.logger = logger
        self.tracks = {}  # track_id -> ObjectTrack
        self.person_tracks = {}  # track_id -> last_position_and_time

    def update_object_track(self, track_id: int, class_name: str, bbox: Tuple[float, float, float, float]) -> None:
        """Update or create an object track"""
        current_time = time.time()

        if track_id not in self.tracks:
            # Create new track
            self.tracks[track_id] = ObjectTrack(track_id, class_name, self.config)
            self.logger.debug(f"Created new object track {track_id} ({class_name})")

        # Update existing track
        self.tracks[track_id].update(bbox, current_time)

    def update_person_track(self, track_id: int, bbox: Tuple[float, float, float, float]) -> None:
        """Update person track for association"""
        current_time = time.time()
        center = GeometryUtils.bbox_center(bbox)
        self.person_tracks[track_id] = (center, current_time, bbox)

    def associate_persons_with_objects(self) -> None:
        """Find associations between persons and objects"""
        current_time = time.time()
        contact_window = self.config.alert.contact_window_seconds
        proximity_threshold = self.config.alert.proximity_threshold_px

        for obj_track in self.tracks.values():
            if obj_track.last_bbox is None:
                continue

            obj_center = GeometryUtils.bbox_center(obj_track.last_bbox)
            associated = False

            # Check proximity to all recent person tracks
            for person_id, (person_center, person_time, person_bbox) in self.person_tracks.items():
                # Only consider recent person detections
                if current_time - person_time > contact_window:
                    continue

                # Calculate distance
                distance = GeometryUtils.euclidean_distance(obj_center, person_center)

                if distance < proximity_threshold:
                    obj_track.associate_with_person(current_time)
                    associated = True
                    self.logger.debug(f"Associated object {obj_track.track_id} with person {person_id} (distance: {distance:.1f}px)")
                    break

            if not associated:
                self.logger.debug(f"Object {obj_track.track_id} not associated with any person")

    def cleanup_old_tracks(self) -> None:
        """Remove old tracks that are no longer active"""
        current_time = time.time()
        max_age = 30.0  # seconds

        # Cleanup object tracks
        inactive_objects = [
            track_id for track_id, track in self.tracks.items()
            if current_time - track.last_seen > max_age
        ]

        for track_id in inactive_objects:
            self.logger.debug(f"Removing inactive object track {track_id}")
            del self.tracks[track_id]

        # Cleanup person tracks
        inactive_persons = [
            track_id for track_id, (_, timestamp, _) in self.person_tracks.items()
            if current_time - timestamp > max_age
        ]

        for track_id in inactive_persons:
            del self.person_tracks[track_id]

    def get_unattended_objects(self) -> List[ObjectTrack]:
        """Get list of objects that should trigger alerts"""
        current_time = time.time()
        return [
            track for track in self.tracks.values()
            if track.should_alert(current_time)
        ]

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get summary of all tracks for debugging"""
        return {
            'total_object_tracks': len(self.tracks),
            'total_person_tracks': len(self.person_tracks),
            'object_details': [track.get_debug_info() for track in self.tracks.values()],
            'stationary_objects': len([t for t in self.tracks.values() if t.stationary_frames > 0]),
            'alerted_objects': len([t for t in self.tracks.values() if t.alerted])
        }


class UnattendedObjectDetector:
    """Main detection system that orchestrates all components"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger("UnattendedDetector", config)

        # Validate dependencies first
        if not validate_dependencies(self.logger):
            raise RuntimeError("Missing required dependencies")

        # Initialize components
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.track_manager = TrackManager(config, self.logger)
        self.alert_manager = AlertManager(config.alert.alert_cooldown_seconds, self.logger)

        # Initialize YOLO and DeepSORT
        self.device = DeviceManager.get_optimal_device(config, self.logger)
        DeviceManager.optimize_torch_settings(self.device, self.logger)

        self.model = None
        self.tracker = None
        self._initialize_models()

        # Frame buffering for event clips
        self.frame_buffer = deque(maxlen=self._calculate_buffer_size())
        self.buffer_timestamps = deque(maxlen=self._calculate_buffer_size())

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_made': 0,
            'tracks_created': 0,
            'alerts_triggered': 0,
            'clips_saved': 0
        }

        # State management
        self.running = True
        self.start_time = time.time()

    def _initialize_models(self) -> None:
        """Initialize ONNX model and DeepSORT tracker"""
        try:
            # Initialize ONNX YOLOv8 with optimized CPU settings
            model_path = self.config.get_model_path().replace('.pt', '.onnx')
            self.logger.info(f"Loading YOLO ONNX model: {model_path}")
            
            # Configure CPU execution provider with optimizations
            cpu_options = DeviceManager.optimize_onnx_cpu_settings()
            
            # Set session options for optimization
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.enable_profiling = False
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            
            # Create optimized session
            providers = [('CPUExecutionProvider', cpu_options)]
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            self.input_name = self.session.get_inputs()[0].name
            self.logger.info("YOLO ONNX model loaded successfully with CPU optimizations")

            # Initialize DeepSORT
            self.tracker = DeepSort(
                max_age=self.config.tracking.max_age,
                n_init=self.config.tracking.n_init,
                max_iou_distance=self.config.tracking.max_iou_distance,
                max_cosine_distance=self.config.tracking.max_cosine_distance,
                nn_budget=self.config.tracking.nn_budget
            )
            self.logger.info("DeepSORT tracker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _calculate_buffer_size(self) -> int:
        """Calculate frame buffer size based on configuration"""
        # Estimate FPS and calculate buffer size
        estimated_fps = 25.0  # Conservative estimate
        buffer_seconds = self.config.video.pre_buffer_seconds + 5  # Extra margin
        return int(buffer_seconds * estimated_fps) + 10

    def _extract_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Extract detections from ONNX YOLOv8 model output, safely handling class indices."""
        try:
            # Preprocess frame
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img)

            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: img})
            preds = outputs[0]
            # Use modular postprocess
            all_detections = yolo_postprocess(
                preds,
                self.config.detection.conf_threshold,
                self.config.detection.coco_classes
            )
            # Only keep detections for monitored object classes or person
            detections = [d for d in all_detections if d['class'].lower() == self.config.detection.person_class_name or d['class'].lower() in [cls.lower() for cls in self.config.detection.object_classes]]
            return detections
        except Exception as e:
            self.logger.error(f"Error extracting detections: {e}")
            return []

    def _filter_detections(self, detections: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Separate object and person detections"""
        object_detections = []
        person_detections = []

        for det in detections:
            class_name = det['class'].lower()

            if class_name == self.config.detection.person_class_name:
                person_detections.append(det)
            elif class_name in [cls.lower() for cls in self.config.detection.object_classes]:
                object_detections.append(det)

        return object_detections, person_detections

    def _prepare_detections_for_tracker(self, detections: List[Dict[str, Any]]) -> List[Tuple]:
        """Prepare detections in format expected by DeepSORT"""
        tracker_detections = []

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class']

            # DeepSORT expects [x1, y1, x2, y2], confidence, class_name
            tracker_detections.append((bbox, conf, class_name))

        return tracker_detections

    def _update_tracks(self, object_detections: List[Dict], person_detections: List[Dict], frame: np.ndarray) -> None:
        """Update all tracks with new detections"""
        try:
            # Combine all detections for tracker
            all_detections = object_detections + person_detections
            tracker_detections = self._prepare_detections_for_tracker(all_detections)

            # Update tracker
            tracks = self.tracker.update_tracks(tracker_detections, frame=frame)

            # Process confirmed tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # Get bounding box
                try:
                    if hasattr(track, 'to_ltrb'):
                        bbox = track.to_ltrb()
                    elif hasattr(track, 'to_tlbr'):
                        bbox = track.to_tlbr()
                    else:
                        continue

                    x1, y1, x2, y2 = map(float, bbox)
                    bbox_tuple = (x1, y1, x2, y2)

                    # Get class name
                    class_name = getattr(track, 'det_class', None) or \
                                getattr(track, 'obj_type', None) or \
                                getattr(track, 'detected_class', None) or "unknown"

                    # Update appropriate track manager
                    if class_name.lower() == self.config.detection.person_class_name:
                        self.track_manager.update_person_track(track_id, bbox_tuple)
                    elif class_name.lower() in [cls.lower() for cls in self.config.detection.object_classes]:
                        self.track_manager.update_object_track(track_id, class_name, bbox_tuple)

                except Exception as e:
                    self.logger.warning(f"Error processing track {track_id}: {e}")
                    continue

            # Update statistics
            self.stats['tracks_created'] += len([t for t in tracks if t.is_confirmed()])

        except Exception as e:
            self.logger.error(f"Error updating tracks: {e}")

    def _process_alerts(self, video_path: str) -> List[Dict[str, Any]]:
        """Process any pending alerts"""
        alerts = []

        # Get objects that should trigger alerts
        unattended_objects = self.track_manager.get_unattended_objects()

        for obj_track in unattended_objects:
            # Check if we can alert (cooldown period)
            if not self.alert_manager.can_alert(obj_track.track_id):
                continue

            # Trigger alert
            alert_data = self.alert_manager.trigger_alert(
                track_id=obj_track.track_id,
                class_name=obj_track.class_name,
                bbox=obj_track.last_bbox,
                reason="unattended",
                video_name=video_path,
                t_unattended=self.config.alert.t_unattended
            )

            alerts.append(alert_data)
            obj_track.alerted = True

            # Save event clip
            if len(self.frame_buffer) > 0:
                self._save_event_clip(obj_track.track_id)

            self.stats['alerts_triggered'] += 1

        return alerts

    def _save_event_clip(self, track_id: int) -> bool:
        """Save event clip for the specified track"""
        try:
            if len(self.frame_buffer) == 0:
                self.logger.warning(f"No frames available for event clip (track {track_id})")
                return False

            # Generate output path
            timestamp = generate_timestamp()
            events_dir = self.config.get_absolute_path(self.config.system.events_dir)
            output_path = os.path.join(events_dir, f"event_{track_id}_{timestamp}.mp4")

            # Convert deque to list for saving
            frames_list = list(self.frame_buffer)

            # Save the clip
            success = VideoProcessor.save_clip(
                frames_list, output_path, 
                self.config.video.buffer_fps, self.logger
            )

            if success:
                self.stats['clips_saved'] += 1
                self.logger.info(f"Event clip saved for track {track_id}: {output_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving event clip for track {track_id}: {e}")
            return False

    def process_frame(self, frame: np.ndarray, video_path: str) -> Dict[str, Any]:
        """Process a single frame"""
        frame_start_time = self.performance_monitor.start_frame()
        current_time = time.time()
        try:
            # Add frame to buffer
            self.frame_buffer.append(frame.copy())
            self.buffer_timestamps.append(current_time)

            # Run ONNX YOLO detection
            detections = self._extract_detections(frame)
            object_detections, person_detections = self._filter_detections(detections)

            # Update tracks
            self._update_tracks(object_detections, person_detections, frame)

            # Associate persons with objects
            self.track_manager.associate_persons_with_objects()

            # Process alerts
            alerts = self._process_alerts(video_path)

            # Cleanup old tracks
            self.track_manager.cleanup_old_tracks()

            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['detections_made'] += len(detections)

            # Calculate processing time
            processing_time = self.performance_monitor.end_frame(frame_start_time)

            # Log performance periodically
            self.performance_monitor.log_performance()

            # Return frame processing results
            return {
                'frame_number': self.stats['frames_processed'],
                'processing_time': processing_time,
                'detections': {
                    'objects': object_detections,
                    'persons': person_detections
                },
                'track_summary': self.track_manager.get_debug_summary(),
                'alerts': alerts,
                'performance': {
                    'fps': self.performance_monitor.get_fps(),
                    'system_stats': self.performance_monitor.get_system_stats()
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return {
                'frame_number': self.stats['frames_processed'],
                'error': str(e),
                'alerts': []
            }

    def run_video(self, video_path: str) -> Dict[str, Any]:
        """Process entire video file"""
        self.logger.info(f"Starting video processing: {video_path}")

        # Validate video format
        if not self.config.validate_video_format(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")

        # Open video
        cap = VideoProcessor.open_video(video_path, self.logger)
        if cap is None:
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            frame_count = 0
            process_every_n = self.config.video.process_every_n_frames

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video reached")
                    break

                frame_count += 1

                # Skip frames if configured to do so
                if frame_count % process_every_n != 0:
                    continue

                # Process frame
                result = self.process_frame(frame, video_path)

                # Handle any errors
                if 'error' in result:
                    self.logger.warning(f"Frame {frame_count} processing error: {result['error']}")

                # Log progress periodically
                if frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    self.logger.info(f"Processed {frame_count} frames in {format_duration(elapsed)}")

        finally:
            cap.release()

        # Generate final report
        return self._generate_final_report(video_path)

    def _generate_final_report(self, video_path: str) -> Dict[str, Any]:
        """Generate final processing report"""
        total_time = time.time() - self.start_time

        report = {
            'video_path': video_path,
            'processing_time': total_time,
            'processing_time_formatted': format_duration(total_time),
            'statistics': self.stats.copy(),
            'final_track_summary': self.track_manager.get_debug_summary(),
            'average_fps': self.performance_monitor.get_fps(),
            'system_performance': self.performance_monitor.get_system_stats()
        }

        self.logger.info("="*60)
        self.logger.info("FINAL PROCESSING REPORT")
        self.logger.info("="*60)
        self.logger.info(f"Video: {os.path.basename(video_path)}")
        self.logger.info(f"Processing Time: {report['processing_time_formatted']}")
        self.logger.info(f"Frames Processed: {self.stats['frames_processed']}")
        self.logger.info(f"Average FPS: {report['average_fps']:.1f}")
        self.logger.info(f"Total Detections: {self.stats['detections_made']}")
        self.logger.info(f"Alerts Triggered: {self.stats['alerts_triggered']}")
        self.logger.info(f"Event Clips Saved: {self.stats['clips_saved']}")
        self.logger.info("="*60)

        return report

    def stop(self) -> None:
        """Stop the detection system"""
        self.running = False
        self.logger.info("Detection system stopped")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unattended Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video video.mp4
  %(prog)s --video /path/to/video.avi --config custom_config.yaml
  %(prog)s --video video.mp4 --debug --model yolov8m.pt
        """
    )

    parser.add_argument(
        '--video', '-v',
        type=str,
        default='video.mp4',
        help='Path to input video file (default: video.mp4)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='YOLO model to use (e.g., yolov8n.pt, yolov8m.pt)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for events and logs'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Force GPU usage (if available)'
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )

    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Detection confidence threshold (0.0-1.0)'
    )

    parser.add_argument(
        '--unattended-time',
        type=float,
        help='Time in seconds before object considered unattended'
    )

    return parser.parse_args()


def setup_configuration(args: argparse.Namespace) -> ConfigManager:
    """Setup configuration from arguments and config file"""

    # Load base configuration
    config = ConfigManager(args.config if args.config else None)

    # Override with command line arguments
    if args.model:
        config.detection.model_size = args.model

    if args.output_dir:
        config.system.events_dir = os.path.join(args.output_dir, 'events')
        config.system.logs_dir = os.path.join(args.output_dir, 'logs')

    if args.debug:
        config.system.log_level = 'DEBUG'

    if args.gpu:
        config.system.enable_gpu = True
        config.detection.device = 'cuda'

    if args.cpu:
        config.system.enable_gpu = False
        config.detection.device = 'cpu'

    if args.conf_threshold:
        config.detection.conf_threshold = args.conf_threshold

    if args.unattended_time:
        config.alert.t_unattended = args.unattended_time

    # Ensure directories exist
    config.ensure_directories()

    return config


def main() -> int:
    """Main application entry point"""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup configuration
        config = setup_configuration(args)

        # Print configuration summary
        config.print_summary()

        # Check if video file exists
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file not found: {args.video}")
            return 1

        print(f"\n🎬 Processing video: {args.video}")
        print("🔄 Initializing detection system...")

        # Initialize detector
        detector = UnattendedObjectDetector(config)

        print("✅ Detection system initialized successfully")
        print("🚀 Starting video processing...\n")

        # Process video
        report = detector.run_video(args.video)

        print("\n✅ Video processing completed successfully!")
        print("📊 Final Report:")
        print(f"   - Processing Time: {report['processing_time_formatted']}")
        print(f"   - Frames Processed: {report['statistics']['frames_processed']}")
        print(f"   - Alerts Triggered: {report['statistics']['alerts_triggered']}")
        print(f"   - Event Clips Saved: {report['statistics']['clips_saved']}")

        if report['statistics']['alerts_triggered'] > 0:
            events_dir = config.get_absolute_path(config.system.events_dir)
            print(f"   - Event clips saved to: {events_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
