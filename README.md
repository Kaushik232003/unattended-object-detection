# Unattended Object Detection System

A production-ready real-time detection system for abandoned objects using YOLOv8 and DeepSORT tracking. The system monitors video feeds to detect when objects are left unattended for configurable periods and generates alerts with event clips.

## 🎯 Features

- **Real-time Object Detection**: Uses YOLOv8 for high-accuracy object detection
- **Multi-object Tracking**: DeepSORT tracking for consistent object identification
- **Person-Object Association**: Intelligent proximity-based association logic
- **Unattended Detection**: Configurable thresholds for abandonment detection
- **Alert System**: Console alerts and automatic event clip saving
- **Multiple Video Formats**: Support for MP4, AVI, MOV, MKV, and more
- **GPU Acceleration**: CUDA and MPS support with CPU fallback
- **Comprehensive Logging**: Detailed logs with performance monitoring
- **Flexible Configuration**: YAML-based configuration management
- **Production Ready**: Error handling, monitoring, and optimization

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended for GPU)
- GPU with CUDA support (optional but recommended)

### Supported Video Formats
- MP4, AVI, MOV, MKV, FLV, WMV, M4V, 3GP

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd unattended_object_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run with default settings (expects video.mp4 in current directory)
python app.py

# Specify video file
python app.py --video /path/to/your/video.mp4

# Enable debug mode
python app.py --video video.mp4 --debug

# Use different YOLO model (for better accuracy)
python app.py --video video.mp4 --model yolov8m.pt
```

### 3. Expected Output

When the system detects an unattended object, you'll see console alerts like:
```
[ALERT] 2025-09-15T12:34:56.789Z Video=video.mp4 object_track=42 class=backpack reason=unattended t_unattended=3.0s
```

Event clips are automatically saved to the `events/` directory.

## 🔧 Configuration

### Configuration File

Create a `config.yaml` file to customize system behavior:

```yaml
detection:
  model_size: "yolov8n.pt"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  conf_threshold: 0.35
  iou_threshold: 0.45
  device: "auto"  # auto, cpu, cuda, mps

alert:
  t_unattended: 3.0  # Seconds before object considered unattended
  confirm_frames: 3  # Consecutive frames to confirm condition
  motion_threshold_px: 2.5  # Pixel movement threshold
  proximity_threshold_px: 120.0  # Person-object proximity in pixels

video:
  process_every_n_frames: 1  # Process every N frames (1=all frames)
  pre_buffer_seconds: 5.0  # Seconds of video before alert
  post_buffer_seconds: 5.0  # Seconds of video after alert

system:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  enable_gpu: true
```

### Command Line Options

```bash
python app.py [OPTIONS]

Options:
  --video, -v PATH          Input video file (default: video.mp4)
  --config, -c PATH         Configuration YAML file
  --model, -m MODEL         YOLO model (yolov8n.pt, yolov8m.pt, etc.)
  --output-dir, -o PATH     Output directory for events and logs
  --debug                   Enable debug logging
  --gpu                     Force GPU usage
  --cpu                     Force CPU usage
  --conf-threshold FLOAT    Detection confidence threshold (0.0-1.0)
  --unattended-time FLOAT   Unattended time threshold in seconds
```

## 📁 Project Structure

```
unattended_object_detection/
├── app.py                    # Main application
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── events/                   # Event clips (auto-created)
├── logs/                     # Log files (auto-created)
├── models/                   # YOLO models (auto-created)
└── config.yaml              # Optional configuration file
```

## 🧠 How It Works

### Detection Pipeline

1. **Video Input**: Reads video frame by frame
2. **Object Detection**: YOLOv8 detects objects and persons in each frame
3. **Object Tracking**: DeepSORT maintains consistent object IDs across frames
4. **Motion Analysis**: Tracks object movement to determine if stationary
5. **Person Association**: Associates objects with nearby persons
6. **Abandonment Detection**: Triggers alerts for unattended stationary objects
7. **Event Recording**: Saves video clips around alert events

### Alert Logic

An object is considered **unattended** when:
- Object is stationary (motion below threshold for N consecutive frames)
- No person detected within proximity threshold for T seconds
- Condition persists for confirmation period

Default thresholds:
- **Unattended Time**: 3.0 seconds
- **Motion Threshold**: 2.5 pixels/frame
- **Proximity Threshold**: 120 pixels
- **Confirmation Frames**: 3 consecutive frames

## 📊 Monitored Object Classes

The system monitors these object types by default:

**Bags & Containers**: backpack, handbag, suitcase, bag
**Electronics**: laptop, cell phone, remote, keyboard, mouse
**Personal Items**: bottle, umbrella, tie, book, clock
**Household Items**: microwave, oven, toaster, blender, bowl

You can modify the monitored classes in `config.py` or your configuration file.

## ⚡ Performance Optimization

### GPU Acceleration

For best performance, ensure proper GPU setup:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA-compatible PyTorch (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Selection

Choose YOLO model based on your hardware:

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|---------|----------|
| yolov8n | Fastest | Good | Low | CPU/Edge devices |
| yolov8s | Fast | Better | Medium | Balanced |
| yolov8m | Medium | High | High | GPU recommended |
| yolov8l | Slow | Higher | Very High | High-end GPU |
| yolov8x | Slowest | Highest | Extreme | Research/Offline |

### Processing Options

```bash
# Process every 2nd frame (2x speed, some accuracy loss)
python app.py --video video.mp4 --config config.yaml
# Then set process_every_n_frames: 2 in config

# Lower resolution for speed
# Set img_size: 416 in config (default: 640)

# Reduce confidence threshold for more detections
python app.py --video video.mp4 --conf-threshold 0.25
```

## 📝 Example Workflows

### Basic Security Monitoring

```bash
# Monitor for abandoned bags in public area
python app.py \
  --video security_camera.mp4 \
  --unattended-time 10.0 \
  --model yolov8m.pt \
  --output-dir /security/alerts/
```

### Airport Baggage Monitoring

```yaml
# config_airport.yaml
detection:
  model_size: "yolov8l.pt"
  conf_threshold: 0.4
  object_classes: ["backpack", "handbag", "suitcase", "bag", "laptop"]

alert:
  t_unattended: 30.0  # 30 seconds for airport
  proximity_threshold_px: 200.0  # Larger area
  confirm_frames: 5
```

```bash
python app.py --video airport_cam.mp4 --config config_airport.yaml
```

### Development and Testing

```bash
# Debug mode with detailed logging
python app.py \
  --video test_video.mp4 \
  --debug \
  --conf-threshold 0.2 \
  --unattended-time 1.0
```

## 🔍 Troubleshooting

### Common Issues

**Error: "Failed to import required libraries"**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Error: "Video file not found"**
```bash
# Check file path and format
python app.py --video /full/path/to/video.mp4
```

**Low FPS/Performance**
```bash
# Try smaller model
python app.py --video video.mp4 --model yolov8n.pt

# Force CPU if GPU issues
python app.py --video video.mp4 --cpu

# Process fewer frames
# Set process_every_n_frames: 3 in config
```

**No GPU detected**
```bash
# Check CUDA installation
nvidia-smi

# Install correct PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Too many false alerts**
```bash
# Increase thresholds
python app.py \
  --video video.mp4 \
  --unattended-time 10.0 \
  --conf-threshold 0.5
```

**Missing alerts**
```bash
# Decrease thresholds
python app.py \
  --video video.mp4 \
  --unattended-time 1.0 \
  --conf-threshold 0.3 \
  --debug
```

### Log Analysis

Check log files for detailed information:
```bash
# View main log
tail -f logs/UnattendedDetector.log

# Debug mode for verbose output
python app.py --video video.mp4 --debug
```

## 🧪 Testing

### Functional Testing

Test with scenarios where:
1. Person places bag and walks away → Should trigger alert
2. Person places bag and stays nearby → Should not trigger alert
3. Person retrieves bag within time limit → Should not trigger alert
4. Moving objects (rolling ball) → Should not trigger alert

### Performance Testing

```bash
# Measure processing speed
time python app.py --video test_video.mp4

# Monitor system resources
python app.py --video video.mp4 --debug
# Check logs for CPU/memory usage
```

## 🔧 Advanced Configuration

### Custom Object Classes

Modify `config.py` or your YAML config:

```yaml
detection:
  object_classes:
    - "backpack"
    - "handbag" 
    - "suitcase"
    - "laptop"
    - "bottle"
    # Add your custom classes here
```

### Alert Cooldown

Prevent alert spam:

```yaml
alert:
  alert_cooldown_seconds: 30.0  # Minimum time between alerts for same object
```

### Video Processing

```yaml
video:
  supported_formats: [".mp4", ".avi", ".mov", ".mkv"]
  buffer_fps: 15.0  # FPS for saved event clips
```

## 📈 Performance Metrics

The system provides comprehensive metrics:

- **Processing FPS**: Real-time processing speed
- **Detection Rate**: Objects detected per frame
- **Track Accuracy**: Successful object tracking
- **Alert Precision**: True vs false positive rate
- **System Resources**: CPU, memory, GPU usage

## 🛡️ Security Considerations

- Event clips may contain personal/private information
- Implement data retention policies
- Consider face blurring for privacy
- Secure storage of alert videos
- Monitor system access and logs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **DeepSORT**: Multi-object tracking algorithm
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Create an issue with detailed description and logs
4. Include system specifications and video details

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Compatibility**: Python 3.8+, CUDA 11.8+
