"""
TensorRT Inference Engine for YOLOv8
Provides high-performance inference using TensorRT engine files
"""

import numpy as np
import tensorrt as trt
from typing import List, Dict, Any, Tuple
import cv2


class TensorRTEngine:
    """TensorRT inference engine for YOLOv8 models"""
    
    def __init__(self, engine_path: str, logger=None):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path: Path to .engine file
            logger: Logger instance for debugging
        """
        self.engine_path = engine_path
        self.logger = logger
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Load engine
        self._load_engine()
        self._allocate_buffers()
        
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Load engine
            with open(self.engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            if self.engine is None:
                raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
                
            self.context = self.engine.create_execution_context()
            if self.logger:
                self.logger.info(f"TensorRT engine loaded successfully: {self.engine_path}")
                
        except Exception as e:
            error_msg = f"Failed to load TensorRT engine: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _allocate_buffers(self):
        """Allocate memory buffers using TensorRT only"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host buffer only
            host_mem = np.empty(size, dtype)
            
            # Use host memory pointer for bindings
            self.bindings.append(int(host_mem.ctypes.data))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'host': host_mem,
                    'shape': self.engine.get_binding_shape(binding),
                    'dtype': dtype,
                    'name': binding
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'shape': self.engine.get_binding_shape(binding),
                    'dtype': dtype,
                    'name': binding
                })
                
        if self.logger:
            self.logger.info(f"Allocated buffers - Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv8 inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size (assuming 640x640 for YOLOv8)
        input_shape = self.inputs[0]['shape']
        if len(input_shape) == 4:  # [batch, channels, height, width]
            target_size = (input_shape[3], input_shape[2])  # (width, height)
        else:
            target_size = (640, 640)  # Default for YOLOv8
            
        img = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Ensure contiguous array
        img = np.ascontiguousarray(img)
        
        return img
    
    def inference(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image using TensorRT only
        
        Args:
            preprocessed_image: Preprocessed image tensor
            
        Returns:
            Raw inference output
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], preprocessed_image.ravel())
        
        # Run synchronous inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Reshape output
        output_shape = self.outputs[0]['shape']
        output = self.outputs[0]['host'].reshape(output_shape)
        
        return output
    
    def postprocess_output(self, output: np.ndarray, conf_threshold: float, coco_classes: List[str]) -> List[Dict[str, Any]]:
        """
        Postprocess TensorRT output to get detections
        
        Args:
            output: Raw model output
            conf_threshold: Confidence threshold
            coco_classes: List of class names
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # YOLOv8 output format: [batch, num_detections, 84]
        # Where 84 = 4 (bbox) + 80 (classes)
        if len(output.shape) == 3:
            predictions = output[0]  # Remove batch dimension
        else:
            predictions = output
            
        # Transpose if needed to get [num_detections, 84]
        if predictions.shape[0] == 84:
            predictions = predictions.T
            
        for pred in predictions:
            # Extract bbox and class scores
            if len(pred) < 84:  # Ensure we have enough elements
                continue
                
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:84]  # 80 classes
            
            # Find maximum confidence class
            max_conf = np.max(class_scores)
            if max_conf < conf_threshold:
                continue
                
            class_id = int(np.argmax(class_scores))
            if class_id >= len(coco_classes):
                continue
                
            # Convert center format to corner format
            x1 = float(x_center - width / 2)
            y1 = float(y_center - height / 2)
            x2 = float(x_center + width / 2)
            y2 = float(y_center + height / 2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_conf),
                'class': coco_classes[class_id],
                'class_id': class_id
            })
            
        return detections
    
    def predict(self, image: np.ndarray, conf_threshold: float, coco_classes: List[str]) -> List[Dict[str, Any]]:
        """
        Complete inference pipeline: preprocess -> inference -> postprocess
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold
            coco_classes: List of class names
            
        Returns:
            List of detection dictionaries
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Inference
        output = self.inference(preprocessed)
        
        # Postprocess
        detections = self.postprocess_output(output, conf_threshold, coco_classes)
        
        return detections
    
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get model input shape (batch, channels, height, width)"""
        return tuple(self.inputs[0]['shape'])
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get model output shape"""
        return tuple(self.outputs[0]['shape'])
    
    def cleanup(self):
        """Clean up TensorRT resources"""
        if self.context:
            del self.context
            self.context = None
        if self.engine:
            del self.engine
            self.engine = None
            
        if self.logger:
            self.logger.info("TensorRT resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


def create_tensorrt_engine(onnx_path: str, engine_path: str, max_batch_size: int = 1, 
                          workspace_size: int = 1 << 30, fp16_mode: bool = True, 
                          logger=None) -> bool:
    """
    Convert ONNX model to TensorRT engine
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        max_batch_size: Maximum batch size
        workspace_size: Workspace size in bytes (default 1GB)
        fp16_mode: Enable FP16 precision
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Create builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX file
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                if logger:
                    logger.error("Failed to parse ONNX file")
                    for i in range(parser.num_errors):
                        logger.error(f"Parser error {i}: {parser.get_error(i)}")
                return False
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            if logger:
                logger.info("Using FP16 precision")
        else:
            if logger:
                logger.info("Using FP32 precision")
        
        # Set optimization profiles for dynamic shapes (if needed)
        profile = builder.create_optimization_profile()
        
        # Get input tensor
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape
        
        # Set profile for dynamic batch size
        min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        opt_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])
        max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])
        
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        if logger:
            logger.info("Building TensorRT engine... This may take several minutes.")
            
        engine = builder.build_engine(network, config)
        
        if engine is None:
            if logger:
                logger.error("Failed to build TensorRT engine")
            return False
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
            
        if logger:
            logger.info(f"TensorRT engine saved to: {engine_path}")
            
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating TensorRT engine: {e}")
        return False