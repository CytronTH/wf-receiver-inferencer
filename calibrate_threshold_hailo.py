import numpy as np
import cv2
import argparse
import time
import os
import glob

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, 
                                InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)
except ImportError:
    print("Error: hailo_platform not found.")
    exit(1)

# Define the Inference Class directly here to make script standalone
class HailoPatchCoreOptimizedFunc:
    def __init__(self, hef_path, device_id='d000', size=320):
        self.hef_path = hef_path
        self.device_id = device_id
        self.size = size
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True
        params.group_id = "shared_hailo_group"
        self.target = VDevice(params=params)
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        print("Hailo Initialized.")

    def infer(self, image):
        # Preprocess
        h, w = self.size, self.size
        if image.shape[0] != h or image.shape[1] != w:
             img_resized = cv2.resize(image, (w, h))
        else:
             img_resized = image
             
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_norm = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        
        input_data = np.expand_dims(img_norm, axis=0).astype(np.float32)
        
        # Run Inference
        with self.network_group.activate(self.network_group_params):
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                input_name = list(self.input_vstreams_params.keys())[0]
                infer_results = infer_pipeline.infer({input_name: input_data})
            
            output_name = list(infer_results.keys())[0]
            normality_map = infer_results[output_name][0]
            
            if len(normality_map.shape) == 3:
                 normality_map = normality_map[:, :, 0]
                 
            anomaly_map = 1.0 - normality_map
            anomaly_map = np.clip(anomaly_map, 0, 1)
            
            return np.max(anomaly_map)

def get_image_files(directory):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", required=True, help="Path to HEF model")
    parser.add_argument("--ok", required=True, help="Path to folder containing OK images")
    parser.add_argument("--ng", required=True, help="Path to folder containing NG images")
    parser.add_argument("--size", type=int, default=320, help="Input size")
    parser.add_argument("--output", default="threshold_graph.jpg", help="Output graph filename")
    args = parser.parse_args()
    
    if not os.path.exists(args.hef):
        print(f"Error: HEF file {args.hef} not found.")
        return

    # Load Model
    try:
        model = HailoPatchCoreOptimizedFunc(args.hef, size=args.size)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get Files
    ok_files = get_image_files(args.ok)
    ng_files = get_image_files(args.ng)
    
    if not ok_files:
        print(f"No images found in OK directory: {args.ok}")
        return
    if not ng_files:
        print(f"No images found in NG directory: {args.ng}")
        return
        
    print(f"\nFound {len(ok_files)} OK images and {len(ng_files)} NG images.")
    print("Starting Inference...")
    
    ok_scores = []
    ng_scores = []
    
    # Process OK
    print("\nProcessing OK images...")
    for f in ok_files:
        img = cv2.imread(f)
        if img is None: continue
        score = model.infer(img)
        ok_scores.append(score)
        print(f"  {os.path.basename(f)}: {score:.4f}")
        
    # Process NG
    print("\nProcessing NG images...")
    for f in ng_files:
        img = cv2.imread(f)
        if img is None: continue
        score = model.infer(img)
        ng_scores.append(score)
        print(f"  {os.path.basename(f)}: {score:.4f}")

    # Statistics
    ok_scores = np.array(ok_scores)
    ng_scores = np.array(ng_scores)
    
    print("\n" + "="*40)
    print("       STATISTICS REPORT       ")
    print("="*40)
    
    print(f"OK Images: {len(ok_scores)}")
    print(f"  Min: {ok_scores.min():.4f}")
    print(f"  Max: {ok_scores.max():.4f}")
    print(f"  Avg: {ok_scores.mean():.4f}")
    print(f"  Std: {ok_scores.std():.4f}")
    
    print("-" * 20)
    
    print(f"NG Images: {len(ng_scores)}")
    print(f"  Min: {ng_scores.min():.4f}")
    print(f"  Max: {ng_scores.max():.4f}")
    print(f"  Avg: {ng_scores.mean():.4f}")
    print(f"  Std: {ng_scores.std():.4f}")
    
    print("="*40)
    
    # Suggest Threshold (F1 Score Maximization)
    print(f"\nGap Analysis:")
    print(f"Max OK Score: {ok_scores.max():.4f}")
    print(f"Min NG Score: {ng_scores.min():.4f}")
    
    # 1. Collect all scores
    labels_ok = np.zeros(len(ok_scores))
    labels_ng = np.ones(len(ng_scores))
    
    all_scores = np.concatenate([ok_scores, ng_scores])
    all_labels = np.concatenate([labels_ok, labels_ng])
    
    # 2. Sweep thresholds
    best_f1 = -1
    best_thresh = 0
    
    # Sort scores to use as candidate thresholds
    sorted_scores = np.sort(np.unique(all_scores))
    # Add midpoints
    thresholds = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0
    thresholds = np.concatenate([[sorted_scores[0]-0.01], thresholds, [sorted_scores[-1]+0.01]])
    
    for t in thresholds:
        preds = (all_scores > t).astype(int)
        
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        # Avoid div by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\n[OPTIMAL THRESHOLD FOUND]: {best_thresh:.4f}")
    print(f"Best F1-Score: {best_f1:.4f}")

    # 3. List Overlaps (Errors at this threshold)
    print("\n" + "="*40)
    print("       OVERLAP / ERROR REPORT       ")
    print("="*40)
    
    print("\nFalse Negatives (NG classified as OK): -> Dangerous!")
    fn_count = 0
    # We need filenames back. Re-iterate or zip
    # Re-iterate NG
    for f in ng_files:
        img = cv2.imread(f)
        if img is None: continue
        s = model.infer(img)
        if s <= best_thresh:
            print(f"  [FN] {os.path.basename(f)}: {s:.4f}")
            fn_count += 1
            
    print(f"Total FN: {fn_count}")
    
    print("\nFalse Positives (OK classified as NG): -> Annoying but safe")
    fp_count = 0
    # Re-iterate OK
    for f in ok_files:
        img = cv2.imread(f)
        if img is None: continue
        s = model.infer(img)
        if s > best_thresh:
            print(f"  [FP] {os.path.basename(f)}: {s:.4f}")
            fp_count += 1
            
    print(f"Total FP: {fp_count}")
    
    # 4. Draw Graph (Histogram) using OpenCV
    print("\nGenerating Histogram Graph...")
    
    # Config
    W, H = 800, 600
    graph_img = np.ones((H, W, 3), dtype=np.uint8) * 255 # White background
    
    # Define bins
    min_val = min(ok_scores.min(), ng_scores.min())
    max_val = max(ok_scores.max(), ng_scores.max())
    # Add padding
    min_val = max(0, min_val - 0.1)
    max_val = min(1.0, max_val + 0.1)
    
    bins = 50
    bin_width = (max_val - min_val) / bins
    
    hist_ok = np.zeros(bins)
    hist_ng = np.zeros(bins)
    
    for s in ok_scores:
        idx = int((s - min_val) / bin_width)
        idx = min(idx, bins-1)
        hist_ok[idx] += 1
        
    for s in ng_scores:
        idx = int((s - min_val) / bin_width)
        idx = min(idx, bins-1)
        hist_ng[idx] += 1
        
    # Normalize height
    max_freq = max(hist_ok.max(), hist_ng.max())
    scale_y = (H - 100) / max_freq
    
    # Draw logic
    start_x = 50
    graph_w = W - 100
    bar_w = int(graph_w / bins)
    
    for i in range(bins):
        # OK Bar (Green)
        h_ok = int(hist_ok[i] * scale_y)
        # NG Bar (Red) - Transparent? No, just stack or side-by-side or outline
        # Let's draw OK filled, NG outline? Or Overlap color?
        h_ng = int(hist_ng[i] * scale_y)
        
        x = start_x + i * bar_w
        base_y = H - 50
        
        # OK (Green)
        if h_ok > 0:
            cv2.rectangle(graph_img, (x, base_y - h_ok), (x + bar_w, base_y), (0, 255, 0), -1) # Green Filled
            
        # NG (Red)
        if h_ng > 0:
            # Draw semi-transparent logic manually or just line
            # Draw Red Rectangle outline thicker
            cv2.rectangle(graph_img, (x, base_y - h_ng), (x + bar_w, base_y), (0, 0, 255), 2) # Red Outline
            
    # Draw Threshold Line
    t_x = start_x + int((best_thresh - min_val) / (max_val - min_val) * graph_w)
    cv2.line(graph_img, (t_x, 50), (t_x, H - 50), (255, 0, 0), 2) # Blue Line
    cv2.putText(graph_img, f"Thresh: {best_thresh:.3f}", (t_x + 5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw Legend
    cv2.rectangle(graph_img, (W - 200, 20), (W - 20, 100), (240, 240, 240), -1)
    cv2.putText(graph_img, "Green: OK", (W - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.putText(graph_img, "Red: NG", (W - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    out_graph = args.output
    cv2.imwrite(out_graph, graph_img)
    print(f"\nSaved graph to {out_graph}")

if __name__ == "__main__":
    main()
