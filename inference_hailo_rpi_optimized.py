import numpy as np
import cv2
import argparse
import time
import os

# Hailo Imports
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, 
                                InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)
except ImportError:
    print("Error: hailo_platform not found. This script is intended to run on the Raspberry Pi with Hailo software installed.")

class HailoPatchCoreOptimized:
    def __init__(self, hef_path, device_id='d000', size=320):
        self.hef_path = hef_path
        self.device_id = device_id
        self.size = size
        
        # Initialize Hailo with Multi-Process Service enabled
        print(f"Initializing Hailo Device {device_id} (Multi-Process)...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True
        params.group_id = "shared_hailo_group"
        self.target = VDevice(params=params)
        
        # Load HEF
        self.hef = HEF(hef_path)
        
        # Configure Network Groups
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Input/Output Stream Params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        print("Hailo Initialized Successfully (Optimized End-to-End Mode).")

    def infer(self, image):
        """
        Runs inference on a single image.
        image: numpy array (H, W, 3) BGR or RGB
        """
        # Preprocess
        h, w = self.size, self.size
        img_resized = cv2.resize(image, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_norm = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        
        # Add batch dim: [1, H, W, C]
        input_data = np.expand_dims(img_norm, axis=0).astype(np.float32)
        
        # Run Inference
        with self.network_group.activate(self.network_group_params):
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                input_name = list(self.input_vstreams_params.keys())[0]
                start_time = time.time()
                
                # Run
                infer_results = infer_pipeline.infer({input_name: input_data})
            
            # Extract Output (Normality Map)
            # keys might be 'normality_map' or 'resnet18/something'
            # Just take the first output
            output_name = list(infer_results.keys())[0]
            normality_map = infer_results[output_name][0] # [H, W, C] or [C, H, W]? 
            # Hailo output is usually NHWC or just flat buffer reshaped.
            # Our model output was (1, 1, 40, 40) -> (1, 40, 40, 1) likely in Hailo/Numpy land
            
            hailo_time = time.time() - start_time
            
            # Post-Process
            # Convert Normality (High=Good) to Anomaly (High=Bad)
            # Since similarity is dot product of normalized vectors, range is [-1, 1].
            # 1.0 = Exact Match.
            # Anomaly Score = 1.0 - Similarity
            
            # Unwrap dimensions
            if len(normality_map.shape) == 3: # (H, W, 1)
                 normality_map = normality_map[:, :, 0]
                 
            anomaly_map = 1.0 - normality_map
            
            # Clip negative values (if any artifacts)
            anomaly_map = np.clip(anomaly_map, 0, 1)
            
            max_score = np.max(anomaly_map)
            
            print(f"Inference Time: Hailo={hailo_time*1000:.2f}ms")
            
            return max_score, anomaly_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", required=True, help="Path to compiled HEF model")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--size", type=int, default=320, help="Input size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly Threshold")
    args = parser.parse_args()
    
    if not os.path.exists(args.hef):
        print("HEF file not found.")
        return
        
    # No bank needed!
    patchcore = HailoPatchCoreOptimized(args.hef, size=args.size)
    
    img = cv2.imread(args.input)
    score, amap = patchcore.infer(img)
    
    print(f"Anomaly Score: {score:.4f}")
    result_status = "NG" if score > args.threshold else "OK"
    print(f"Result: {result_status}")
    
    # Visualization
    # Resize map to image size
    heatmap = cv2.resize(amap, (img.shape[1], img.shape[0]))
    
    # Normalize for vis (0-1 -> 0-255)
    # Use global normalization or local? 
    # For consistent visual feedback, 0-1 range is best logic strictly. 
    # But usually we normalize 0-Max for contrast.
    # Let's normalize 0-Max for visibility
    vis_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    vis_uint8 = (vis_norm * 255).astype(np.uint8)
    
    heatmap_color = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    cv2.putText(overlay, f"{result_status} ({score:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite("result_hailo_optimized.jpg", overlay)
    print("Saved result_hailo_optimized.jpg")

if __name__ == "__main__":
    main()
