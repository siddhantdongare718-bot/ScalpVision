# cluster point
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt

import numpy as np
import cv2
from tqdm import tqdm
import json
import os

# --- PATHS ---
points_path = '/content/ScalpVision/train_seg_points.json'
sam_checkpoint = "/content/ScalpVision/sam_vit_h_4b8939.pth"
u2net_results_dir = '/content/ScalpVision/u2net_results/'
sam_results_dir = '/content/ScalpVision/sam_results/'
input_images_dir = '/content/ScalpVision/input_images/'

# Create output directory if it doesn't exist
os.makedirs(sam_results_dir, exist_ok=True)

with open(points_path,'r') as f:
    points = json.load(f)

model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for full_name in tqdm(points.keys()):
    name = full_name.split('.')[0]
    
    # Handle filename extensions (try .png for mask, .jpg/.png for input)
    sample_points = points.get(full_name, [])
    
    if len(sample_points) == 0:
        # Fallback to U2Net result if no points found
        image_path = os.path.join(u2net_results_dir, f'{name}.png')
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            cv2.imwrite(os.path.join(sam_results_dir, f'{name}.jpg'), image)
        else:
            print(f"Warning: Could not find U2Net result for {name}")
    else:
        tmp = np.array(sample_points)
        # Filter valid points
        if tmp.size > 0:
            tmp = tmp[tmp.min(axis=1) > 0]

        if tmp.size == 0:
             # Fallback if points exist but are invalid
            image_path = os.path.join(u2net_results_dir, f'{name}.png')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                cv2.imwrite(os.path.join(sam_results_dir, f'{name}.jpg'), image)
            continue

        rand_idx = np.random.choice(len(tmp), max(1, len(tmp)//2), replace=False)

        # Load Input Image
        # Check for both jpg and png extensions for input
        img_path_jpg = os.path.join(input_images_dir, f'{name}.jpg')
        img_path_png = os.path.join(input_images_dir, f'{name}.png')
        
        if os.path.exists(img_path_jpg):
            image = cv2.imread(img_path_jpg)
        elif os.path.exists(img_path_png):
            image = cv2.imread(img_path_png)
        else:
            print(f"Error: Input image for {name} not found.")
            continue

        # --- KEY FIX: Get actual image dimensions ---
        h, w, _ = image.shape
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        neg_list = []
        while len(neg_list) < 10:
            # --- KEY FIX: Use actual width (w) and height (h) ---
            xy = [np.random.randint(w), np.random.randint(h)]
            
            # Simple check if point is far from hair points (optional refinement)
            # For now, just ensuring it's not in the exact positive list
            # Converting to tuple for faster lookup if tmp is large, but list check is okay for small N
            is_close = False
            for p in tmp:
                if np.linalg.norm(np.array(xy) - p) < 5: # 5 pixel buffer
                    is_close = True
                    break
            if not is_close:
                neg_list.append(xy)
                
        neg_arr = np.array(neg_list) # scalp background points

        input_point = tmp[rand_idx] # hair points
        
        final_point = np.append(input_point, neg_arr).reshape(-1,2)
        input_label = np.array([0] * len(input_point) + [1]*len(neg_arr))

        masks, scores, logits = predictor.predict(
            point_coords=final_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # --- KEY FIX: Remove hardcoded reshape. Use mask shape directly ---
        best_mask_idx = np.argmax(scores)
        sam_mask = masks[best_mask_idx] # Shape is already (H, W)
        
        binary_map = np.where(sam_mask > 0, 0, 255).astype(np.uint8)

        # Get rid of noises (ex. big white spots that are not hair)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:, cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        
        for i in range(0, nlabels - 1):
            if areas[i] >= 400:   # Keep large enough areas
                result[labels == i + 1] = 255
                
        save_path = os.path.join(sam_results_dir, name + '.jpg')
        cv2.imwrite(save_path, result)
