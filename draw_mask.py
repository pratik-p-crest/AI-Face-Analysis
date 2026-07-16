import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import sys

def get_robust_landmarks(labels, contour, person_mask, h, w):
    M = cv2.moments(person_mask)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])

    # Find Nose
    nose_mask = (labels == 2).astype(np.uint8)
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(nose_mask, connectivity=8)
    if num_labels <= 1:
        return None
    
    largest_nose = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    nose_mask_clean = (comp_labels == largest_nose).astype(np.uint8)
    nose_coords = np.column_stack(np.where(nose_mask_clean > 0))
    if len(nose_coords) == 0:
        return None
        
    temp_nose_mean_x = int(np.mean(nose_coords[:, 1]))
    facing_right = temp_nose_mean_x > cx

    nose_idx = np.argmax(nose_coords[:, 1]) if facing_right else np.argmin(nose_coords[:, 1])
    nose_y, nose_x = nose_coords[nose_idx]
    nose_bottom_y = np.max(nose_coords[:, 0])

    # Get Skin Mask for fallbacks
    skin_mask = (labels == 1).astype(np.uint8)
    skin_coords = np.column_stack(np.where(skin_mask > 0))
    if len(skin_coords) > 0:
        skin_left_x = np.min(skin_coords[:, 1])
        skin_right_x = np.max(skin_coords[:, 1])
        skin_width = skin_right_x - skin_left_x
    else:
        skin_left_x = 0
        skin_right_x = w
        skin_width = w

    # Find Ear
    ear_mask = ((labels == 7) | (labels == 8)).astype(np.uint8)
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(ear_mask, connectivity=8)
    
    valid_ear_found = False
    if num_labels > 1:
        valid_ears = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 20:
                valid_ears.append(i)
        
        if valid_ears:
            best_ear = valid_ears[0]
            best_ear_x = stats[best_ear, cv2.CC_STAT_LEFT]
            for i in valid_ears:
                x = stats[i, cv2.CC_STAT_LEFT]
                if facing_right:
                    if x < best_ear_x:
                        best_ear = i
                        best_ear_x = x
                else:
                    if x > best_ear_x:
                        best_ear = i
                        best_ear_x = x

            ear_mask_clean = (comp_labels == best_ear).astype(np.uint8)
            ear_coords = np.column_stack(np.where(ear_mask_clean > 0))
            
            temp_ear_front_x = np.max(ear_coords[:, 1]) if facing_right else np.min(ear_coords[:, 1])
            
            # Reject if ear is too close to nose (likely misclassified cheek/eye)
            if abs(nose_x - temp_ear_front_x) > skin_width * 0.3:
                valid_ear_found = True
                ear_mean_x = int(np.mean(ear_coords[:, 1]))
                ear_front_x = temp_ear_front_x

    if not valid_ear_found:
        # Fallback to leftmost/rightmost skin edge if ear not found
        ear_front_x = skin_left_x if facing_right else skin_right_x
        ear_mean_x = ear_front_x

    # Find Mouth
    lips_mask = ((labels == 9) | (labels == 10) | (labels == 11)).astype(np.uint8)
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(lips_mask, connectivity=8)
    
    valid_lips = []
    if num_labels > 1:
        for i in range(1, num_labels):
            # Centroid of lips must be below the nose! (filter out misclassified hair/eyes)
            if centroids[i, 1] > nose_y:
                valid_lips.append(i)
                
    if valid_lips:
        best_lips = max(valid_lips, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        lips_mask_clean = (comp_labels == best_lips).astype(np.uint8)
        lips_coords = np.column_stack(np.where(lips_mask_clean > 0))
        mouth_bottom_y = np.max(lips_coords[:, 0])
    else:
        # Fallback if no lips found
        mouth_bottom_y = nose_bottom_y + int(abs(nose_x - ear_mean_x) * 0.15)

    # Estimate Chin Y using standard facial proportions:
    # The distance from nose to chin is proportional to distance from nose to mouth.
    chin_y = int(mouth_bottom_y + (mouth_bottom_y - nose_y) * 0.8)

    return {
        'facing_right': facing_right,
        'ear_front_x': ear_front_x,
        'nose_x': nose_x,
        'nose_y': nose_y,
        'chin_y': chin_y
    }


def draw_geometric_mask(image_path, out_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading face-parsing models on {device}...")
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    print("Running face parsing...")
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=pil_image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Create mask of the entire person
    person_mask = np.isin(labels, list(range(1, 18))).astype(np.uint8)
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: Could not find person contour.")
        return
    contour = max(contours, key=cv2.contourArea).squeeze()

    landmarks = get_robust_landmarks(labels, contour, person_mask, h, w)
    if not landmarks: 
        print("Error: Could not find necessary landmarks.")
        return

    facing_right = landmarks['facing_right']
    ear_front_x = landmarks['ear_front_x']
    nose_x = landmarks['nose_x']
    nose_y = landmarks['nose_y']
    chin_y = landmarks['chin_y']

    # Trapezoid geometry
    top_y = nose_y
    bottom_y = chin_y
    top_back_x = ear_front_x
    bottom_back_x = ear_front_x
    
    if facing_right:
        top_front_x = ear_front_x + int(0.70 * (nose_x - ear_front_x))
        bottom_front_x = ear_front_x + int(0.60 * (nose_x - ear_front_x))
    else:
        top_front_x = ear_front_x - int(0.70 * (ear_front_x - nose_x))
        bottom_front_x = ear_front_x - int(0.60 * (ear_front_x - nose_x))
        
    pts = np.array([
        [top_back_x, top_y],
        [top_front_x, top_y],
        [bottom_front_x, bottom_y],
        [bottom_back_x, bottom_y]
    ], np.int32)
    
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 1)
    
    final_jaw_mask = poly_mask
    
    bg_image = np.ones_like(image_rgb) * 255
    cutout_image = np.where(person_mask[..., None], image_rgb, bg_image)
    overlay = cutout_image.copy()
    overlay[final_jaw_mask == 1] = [170, 170, 170]
    
    alpha = 0.55
    final_image = cv2.addWeighted(overlay, alpha, cutout_image, 1 - alpha, 0)
    
    cv2.imwrite(out_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"Successfully saved output to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python draw_mask.py <input_image_path> <output_image_path>")
    else:
        draw_geometric_mask(sys.argv[1], sys.argv[2])
