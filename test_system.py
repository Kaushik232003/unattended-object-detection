#!/usr/bin/env python3
"""
Simple test script for the Unattended Object Detection System
Tests basic functionality without requiring actual video files
"""

import sys
import os
import tempfile
import numpy as np
import cv2

# Add project directory to path
sys.path.insert(0, os.path.dirname(__file__))

def create_test_video(output_path: str, duration: int = 10, fps: int = 30) -> bool:
    """Create a simple test video with moving objects"""
    try:
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Failed to create video writer for {output_path}")
            return False

        total_frames = duration * fps

        for frame_idx in range(total_frames):
            # Create a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add some moving elements (simulate person and objects)
            time_ratio = frame_idx / total_frames

            # Simulate a person walking
            person_x = int(50 + time_ratio * 400)
            person_y = int(height * 0.6)
            cv2.rectangle(frame, (person_x, person_y), (person_x + 80, person_y + 120), (0, 255, 0), -1)

            # Simulate a stationary bag
            bag_x, bag_y = 300, 350
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 40, bag_y + 30), (255, 0, 0), -1)

            # Add frame number text
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        print(f"Created test video: {output_path} ({total_frames} frames)")
        return True

    except Exception as e:
        print(f"Error creating test video: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    required_modules = [
        ('config', 'ConfigManager'),
        ('utils', 'Logger'),
        ('utils', 'GeometryUtils'),
        ('cv2', None),
        ('numpy', None)
    ]

    import_errors = []

    for module_name, class_name in required_modules:
        try:
            module = __import__(module_name)
            if class_name:
                getattr(module, class_name)
            print(f"  ✓ {module_name}" + (f".{class_name}" if class_name else ""))
        except ImportError as e:
            error_msg = f"  ❌ {module_name}" + (f".{class_name}" if class_name else "") + f": {e}"
            print(error_msg)
            import_errors.append(error_msg)

    return len(import_errors) == 0

def test_configuration():
    """Test configuration management"""
    print("\nTesting configuration...")

    try:
        from config import ConfigManager

        # Test default configuration
        config = ConfigManager()
        print(f"  ✓ Default config loaded")
        print(f"    - YOLO model: {config.detection.model_size}")
        print(f"    - Unattended time: {config.alert.t_unattended}s")
        print(f"    - Video formats: {len(config.video.supported_formats)} supported")

        # Test video format validation
        test_formats = ['.mp4', '.avi', '.xyz', '.mov']
        for fmt in test_formats:
            valid = config.validate_video_format(f"test{fmt}")
            status = "✓" if valid else "❌"
            print(f"    {status} Format {fmt}: {'supported' if valid else 'not supported'}")

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")

    try:
        from utils import GeometryUtils, MotionAnalyzer

        # Test geometry utilities
        bbox1 = (10, 10, 50, 50)
        bbox2 = (30, 30, 70, 70)

        center = GeometryUtils.bbox_center(bbox1)
        area = GeometryUtils.bbox_area(bbox1)
        iou = GeometryUtils.bbox_iou(bbox1, bbox2)
        distance = GeometryUtils.euclidean_distance((25, 25), (50, 50))

        print(f"  ✓ Geometry calculations:")
        print(f"    - Center: {center}")
        print(f"    - Area: {area}")
        print(f"    - IoU: {iou:.3f}")
        print(f"    - Distance: {distance:.1f}")

        # Test motion analyzer
        motion = MotionAnalyzer()
        motion.add_position((100, 100), 1.0)
        motion.add_position((101, 101), 2.0)
        motion.add_position((102, 102), 3.0)

        energy = motion.calculate_motion_energy()
        stationary = motion.is_stationary()

        print(f"  ✓ Motion analysis:")
        print(f"    - Energy: {energy:.3f}")
        print(f"    - Stationary: {stationary}")

        return True

    except Exception as e:
        print(f"  ❌ Utilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 UNATTENDED OBJECT DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)

    # Test results
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Utilities Test", test_utilities)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for use.")

        # Offer to create test video
        print("\nOptional: Create test video for manual testing?")
        response = input("Create test video? (y/n): ").lower().strip()

        if response in ['y', 'yes']:
            test_video_path = "test_video.mp4"
            if create_test_video(test_video_path):
                print(f"\nTest video created: {test_video_path}")
                print("You can now run:")
                print(f"  python app.py --video {test_video_path} --debug")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
