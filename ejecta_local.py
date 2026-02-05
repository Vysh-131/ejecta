import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION CLASS ---
class Config:
    # Paths
    MODEL_NORM = 'n.pt'
    MODEL_INV = 'inv.pt'
    
    # Slicing Parameters (The "No Loss" Logic)
    # We chop large images into these chunk sizes
    SLICE_SIZE = 640       
    OVERLAP_RATIO = 0.25   # 25% overlap prevents cutting craters in half at edges
    
    # Robustness
    CONF_THRES = 0.25      # Minimum confidence
    IOU_THRES = 0.4        # Intersection over Union for merging duplicates
    USE_TTA = True         # Test Time Augmentation (Slower but more accurate)

class EjectaDetector:
    def __init__(self):
        print("Initializing")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using Device: {self.device}")
        
        # Load Models Once (Singleton Pattern)
        try:
            self.model_norm = YOLO(Config.MODEL_NORM).to(self.device)
            self.model_inv = YOLO(Config.MODEL_INV).to(self.device)
            print(" Models loaded.")
        except Exception as e:
            print(f" Could not load models. {e}")
            raise e

    def _smart_slice_inference(self, full_image, model, is_inverted=False):
        """
        The Core Logic: Slices a huge image, detects, and restitches coordinates.
        Does NOT resize the original, so zero detail is lost.
        """
        img_h, img_w = full_image.shape[:2]
        
        # If image is small enough, just run it directly
        if img_h <= Config.SLICE_SIZE and img_w <= Config.SLICE_SIZE:
            return self._run_single_inference(full_image, model, is_inverted, 0, 0)

        step_x = int(Config.SLICE_SIZE * (1 - Config.OVERLAP_RATIO))
        step_y = int(Config.SLICE_SIZE * (1 - Config.OVERLAP_RATIO))
        
        all_detections = []

        # Sliding Window Loop
        for y in range(0, img_h, step_y):
            for x in range(0, img_w, step_x):
                # Calculate coords
                y2 = min(y + Config.SLICE_SIZE, img_h)
                x2 = min(x + Config.SLICE_SIZE, img_w)
                
                # Adjust start if we hit the edge (to keep tile size constant)
                y1 = max(0, y2 - Config.SLICE_SIZE)
                x1 = max(0, x2 - Config.SLICE_SIZE)

                # Crop Tile
                tile = full_image[y1:y2, x1:x2]
                
                # --- INFERENCE ON TILE ---
                # We handle the inversion logic here if needed
                if is_inverted:
                    tile_input = cv2.bitwise_not(tile)
                else:
                    tile_input = tile

                # Run YOLO (With TTA if enabled)
                # augment=True turns on the rotation/flip checks automatically!
                results = model(tile_input, conf=Config.CONF_THRES, verbose=False, augment=Config.USE_TTA)[0]

                # Translate coordinates from Tile-Space to Global-Space
                for box in results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Offset by the tile's position
                    global_box = [
                        bx1 + x1, by1 + y1, 
                        bx2 + x1, by2 + y1, 
                        conf
                    ]
                    all_detections.append(global_box)

        return np.array(all_detections)

    def _run_single_inference(self, img, model, is_inverted, offset_x=0, offset_y=0):
        """Helper for small images or web snippets"""
        if is_inverted: img = cv2.bitwise_not(img)
        results = model(img, conf=Config.CONF_THRES, verbose=False, augment=Config.USE_TTA)[0]
        
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append([x1+offset_x, y1+offset_y, x2+offset_x, y2+offset_y, float(box.conf[0])])
        return np.array(dets)

    def _merge_detections(self, boxes_norm, boxes_inv):
        """
        Merges results from both models and removes overlapping duplicates.
        """
        # Combine lists
        all_boxes = []
        labels = [] # 0 for Norm, 1 for Inv
        
        if len(boxes_norm) > 0:
            all_boxes.append(boxes_norm)
            labels.extend([0] * len(boxes_norm))
            
        if len(boxes_inv) > 0:
            all_boxes.append(boxes_inv)
            labels.extend([1] * len(boxes_inv))
            
        if not all_boxes: return [], []

        all_boxes = np.vstack(all_boxes)
        
        # Apply NMS (Non-Maximum Suppression)
        # We use OpenCV's built-in NMS which is fast and standard
        bboxes = all_boxes[:, :4].tolist()
        scores = all_boxes[:, 4].tolist()
        
        indices = cv2.dnn.NMSBoxes(bboxes, scores, Config.CONF_THRES, Config.IOU_THRES)
        
        final_boxes = []
        final_sources = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(all_boxes[i])
                final_sources.append("Normal" if labels[i] == 0 else "Inverted")
                
        return final_boxes, final_sources

    # --- PUBLIC API FUNCTIONS ---

    def analyze_single_image(self, image_array):
        """
        WEB APP HOOK: Call this function from your future Flask/Streamlit app.
        Input: Numpy array (OpenCV Image)
        Output: Processed Image (with boxes), List of Detections (JSON-ready)
        """
        # 1. Run Smart Slicing on both streams
        dets_norm = self._smart_slice_inference(image_array, self.model_norm, is_inverted=False)
        dets_inv = self._smart_slice_inference(image_array, self.model_inv, is_inverted=True)
        
        # 2. Merge Results
        final_boxes, sources = self._merge_detections(dets_norm, dets_inv)
        
        # 3. Draw Results
        result_img = image_array.copy()
        json_results = []
        
        for box, source in zip(final_boxes, sources):
            x1, y1, x2, y2, conf = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            color = (0, 255, 0) if source == "Normal" else (0, 0, 255)
            
            # Robust Drawing: Thicker lines for large images
            thickness = max(2, int(min(image_array.shape[:2]) / 500))
            
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # Smart Label Placement (Inside if near top edge)
            label = f"{source} {conf:.2f}"
            t_y = y1 - 5 if y1 - 5 > 10 else y1 + 20
            cv2.putText(result_img, label, (x1, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness//2)
            
            json_results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "model": source
            })
            
        return result_img, json_results

    def process_local_folder(self, input_dir, output_dir):
        """
        LOCAL MODE: For batch processing files on your laptop.
        """
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        
        print(f"🚀 Found {len(files)} images. Starting Robust Detection...")
        
        for idx, f in enumerate(files):
            img_path = os.path.join(input_dir, f)
            print(f"   [{idx+1}/{len(files)}] Processing {f}...", end='\r')
            
            # Read Image
            # Note: For TIFs, we might need special handling, but cv2 usually handles 8-bit TIF
            img = cv2.imread(img_path)
            if img is None: continue
            
            # --- CALL THE CORE API ---
            processed_img, _ = self.analyze_single_image(img)
            
            # Save
            save_path = os.path.join(output_dir, f"detected_{f}")
            cv2.imwrite(save_path, processed_img)
            
        print("\nBatch Processing Complete")

# --- ENTRY POINT ---
if __name__ == "__main__":
    INPUT_FOLDER = "input"
    OUTPUT_FOLDER = "output"
    
    # 1. Instantiate the Engine
    engine = EjectaDetector()
    
    # 2. Run Local Batch
    engine.process_local_folder(INPUT_FOLDER, OUTPUT_FOLDER)