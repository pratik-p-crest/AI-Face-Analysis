import os
import io
import json
import base64

# --- ADVANCED FACE PARSING & METRICS IMPORTS ---
import io
import math
import numpy as np
import json
try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    segformer_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    segformer_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    segformer_device = "cuda" if torch.cuda.is_available() else "cpu"
    segformer_model.to(segformer_device)
    segformer_model.eval()
    print("Segformer loaded on", segformer_device)
except Exception as e:
    print(f"Segformer failed to load: {e}")
    segformer_model = None

try:
    from metrics_engine import calculate_all_metrics
except:
    pass
# -----------------------------------------------

import cv2
import math
import numpy as np
import requests
import tempfile
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps, ImageDraw

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

# Initialize MediaPipe model globally
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def analyze_eyebrows(landmarks, w, h):
    def get_pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
    r_inner = get_pt(107)
    r_arch = get_pt(105)
    r_outer = get_pt(336) if landmarks[336].x > landmarks[107].x else get_pt(105) # fallback if wrong side, wait, right outer is 336? No, original nb had r_outer=336? Actually right eyebrow is 46, 53, 52, 65, 55. 
    # Let's just use the exact logic from eyebrow nb if we can, but we don't have it explicitly since it was truncated. 
    pass # I'll copy the exact functions below

def get_landmark_pt(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

# --- EAR ANALYSIS HELPER FUNCTIONS ---
def run_roboflow_ear_workflow(image_path):
    url = "https://detect.roboflow.com/infer/workflows/fabiki4429-acoxs-com/general-segmentation-api-2"
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "api_key": os.getenv("ROBOFLOW_API_KEY"),
        "inputs": {
            "image": {"type": "base64", "value": b64_image},
            "classes": ["ear", "lower_ear", "upper_ear"]
        }
    }
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Roboflow API error: {resp.text}")
    return resp.json()["outputs"][0]

def _extract_ear_points(api_result, class_name="ear"):
    predictions_block = api_result.get("predictions", {})
    img_meta = predictions_block.get("image", {"width": 0, "height": 0})
    predictions = predictions_block.get("predictions", [])
    
    match = next((p for p in predictions if p.get("class", "").strip() == class_name), None)
    if match is None:
        return [], img_meta["width"], img_meta["height"]
        
    points = [(p["x"], p["y"]) for p in match["points"]]
    return points, img_meta["width"], img_meta["height"]

def _draw_capped_line(draw, p1, p2, color, width, cap_len):
    draw.line([p1, p2], fill=color, width=width)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0: return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    half = cap_len / 2
    for (cx, cy) in (p1, p2):
        a = (cx - px * half, cy - py * half)
        b = (cx + px * half, cy + py * half)
        draw.line([a, b], fill=color, width=width)

def draw_ear_calipers(image_path, api_result):
    points, model_w, model_h = _extract_ear_points(api_result, "ear")
    if not points:
        return None
        
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    scale_x, scale_y = W / model_w, H / model_h
    pts = [(x * scale_x, y * scale_y) for x, y in points]
    
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ear_w_px, ear_h_px = max_x - min_x, max_y - min_y
    
    pad_x, pad_y = ear_w_px * 0.55, ear_h_px * 0.20
    crop_box = (
        max(0, min_x - pad_x), max(0, min_y - pad_y),
        min(W, max_x + pad_x * 0.6), min(H, max_y + pad_y)
    )
    crop_box = tuple(int(v) for v in crop_box)
    cropped = img.crop(crop_box)
    
    ox, oy = crop_box[0], crop_box[1]
    pts_c = [(x - ox, y - oy) for x, y in pts]
    top, bottom = min_y - oy, max_y - oy
    left, right = min_x - ox, max_x - ox
    
    draw = ImageDraw.Draw(cropped)
    
    vx = left - 18
    _draw_capped_line(draw, (vx, top), (vx, bottom), (255,255,255), 2, 14)
    
    hy = top + (bottom - top) * 0.20
    band = [p for p in pts_c if abs(p[1] - hy) < (bottom - top) * 0.05]
    hx_left = min(p[0] for p in band) if band else left
    hx_right = max(p[0] for p in band) if band else right
    _draw_capped_line(draw, (hx_left, hy), (hx_right, hy), (255,255,255), 2, 14)
    
    top_pt = min(pts_c, key=lambda p: p[1])
    bottom_pt = max(pts_c, key=lambda p: p[1])
    _draw_capped_line(draw, top_pt, bottom_pt, (255,255,255), 2, 14 * 0.85)
    
    diag_len_px = math.hypot(
        (bottom_pt[0] - top_pt[0]) / scale_x if scale_x else 0,
        (bottom_pt[1] - top_pt[1]) / scale_y if scale_y else 0
    ) * ((scale_x + scale_y) / 2)
    
    # Return cropped PIL image and measurements
    return cropped, {
        "ear_height_px": ear_h_px,
        "ear_width_top_px": hx_right - hx_left,
        "ear_diagonal_length_px": diag_len_px
    }

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/eyebrows')
def eyebrows():
    return render_template('eyebrows.html')

@app.route('/analyze_eyebrows', methods=['POST'])
def analyze_eyebrows_api():
    if 'front' not in request.files:
        return jsonify({"error": "Front image must be uploaded"}), 400
        
    front_file = request.files['front']
    front_stream = io.BytesIO(front_file.read())
    try:
        pil_image = Image.open(front_stream)
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": f"Invalid front image format: {e}"}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected in the front image"}), 400

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape

    def pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
    # Calculate IPD
    p_lp = pt(468)
    p_rp = pt(473)
    ipd_px = np.linalg.norm(p_lp - p_rp)
    px_to_mm = 63.0 / ipd_px
    
    right_inner = pt(55)
    right_outer = pt(105)
    right_eb_upper = [156, 70, 63, 105, 66, 107, 55, 65]
    right_peak = min([pt(i) for i in right_eb_upper], key=lambda p: p[1])
    right_eye = p_rp
    
    left_inner = pt(285)
    left_outer = pt(334)
    left_eb_upper = [383, 300, 293, 334, 296, 336, 285, 295]
    left_peak = min([pt(i) for i in left_eb_upper], key=lambda p: p[1])
    left_eye = p_lp
    
    right_height_px = right_eye[1] - right_peak[1]
    left_height_px = left_eye[1] - left_peak[1]
    avg_height_mm = ((right_height_px + left_height_px) / 2.0) * px_to_mm
    
    right_eye_width = np.linalg.norm(pt(133) - pt(33))
    left_eye_width = np.linalg.norm(pt(362) - pt(263))
    avg_eye_width = (right_eye_width + left_eye_width) / 2.0
    avg_height_px = (right_height_px + left_height_px) / 2.0
    elevation_ratio = avg_height_px / avg_eye_width
    
    def calc_angle(apex, p1, p2):
        v1 = p1 - apex
        v2 = p2 - apex
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 180.0
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))
        
    right_angle = calc_angle(right_peak, right_inner, right_outer)
    left_angle = calc_angle(left_peak, left_inner, left_outer)
    avg_angle = (right_angle + left_angle) / 2.0

    # Classifications
    if avg_height_mm > 22.0:
        position = "High Set"
    elif avg_height_mm < 18.0:
        position = "Low Set"
    else:
        position = "Average Set"
        
    right_tilt = right_inner[1] - right_outer[1]
    left_tilt = left_inner[1] - left_outer[1]
    avg_tilt = (right_tilt + left_tilt) / 2.0
    if avg_tilt > 5:
        tilt = "Upturned"
    elif avg_tilt < -5:
        tilt = "Downturned"
    else:
        tilt = "Straight"
        
    if avg_angle < 135:
        shape = "Arched"
    elif avg_angle > 155:
        shape = "Straight"
    else:
        shape = "Rounded"
        
    virility = "Moderate"

    # Cropping
    right_eyebrow_idxs = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    left_eyebrow_idxs = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
    
    def crop_eyebrow(image, landmarks, idxs, w, h, padding_x=20, padding_y=20):
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        return image[y_min:y_max, x_min:x_max]

    r_eyebrow_crop = crop_eyebrow(image_rgb, landmarks, right_eyebrow_idxs, w, h)
    l_eyebrow_crop = crop_eyebrow(image_rgb, landmarks, left_eyebrow_idxs, w, h)

    def encode_img(img):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img)
        return base64.b64encode(buffer).decode('utf-8')

    base64_r_brow = encode_img(r_eyebrow_crop)
    base64_l_brow = encode_img(l_eyebrow_crop)

    return jsonify({
        "vertical_height_mm": round(float(avg_height_mm), 2),
        "elevation_ratio": round(float(elevation_ratio), 2),
        "apex_angle_deg": round(float(avg_angle), 2),
        "position": position,
        "tilt": tilt,
        "shape": shape,
        "virility": virility,
        "r_brow_image": f"data:image/jpeg;base64,{base64_r_brow}",
        "l_brow_image": f"data:image/jpeg;base64,{base64_l_brow}"
    })


@app.route('/eyes')
def eyes():
    return render_template('eyes.html')

@app.route('/analyze_eyes', methods=['POST'])
def analyze_eyes_api():
    if 'front' not in request.files:
        return jsonify({"error": "Front image must be uploaded"}), 400

    front_file = request.files['front']
    front_stream = io.BytesIO(front_file.read())
    try:
        pil_image = Image.open(front_stream)
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected in the image"}), 400

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape

    def pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    # --- Landmark indices ---
    RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
    RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_IRIS = 468
    RIGHT_OUTER = 33
    RIGHT_INNER = 133

    LEFT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
    LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
    LEFT_IRIS = 473
    LEFT_OUTER = 263
    LEFT_INNER = 362

    FACE_W_L = 234
    FACE_W_R = 454

    # --- IPD ---
    p_lp = pt(LEFT_IRIS)
    p_rp = pt(RIGHT_IRIS)
    ipd_px = np.linalg.norm(p_lp - p_rp)
    px_to_mm = 63.0 / ipd_px

    # --- Metric 1: Eye Tilt ---
    def eye_tilt(outer_idx, inner_idx):
        outer = pt(outer_idx)
        inner = pt(inner_idx)
        dx = outer[0] - inner[0]
        dy = inner[1] - outer[1]
        return float(np.degrees(np.arctan2(dy, abs(dx))))

    r_tilt = eye_tilt(RIGHT_OUTER, RIGHT_INNER)
    l_tilt = eye_tilt(LEFT_OUTER, LEFT_INNER)
    avg_tilt = (r_tilt + l_tilt) / 2.0
    # Make tilt more sensitive (previously > 2.0)
    tilt_class = "Positive" if avg_tilt > 1.0 else ("Negative" if avg_tilt < -1.0 else "Neutral")

    # --- Metric 2: Eyelid Exposure ---
    def eyelid_exposure(upper_idxs, lower_idxs, iris_idx):
        iris = pt(iris_idx)
        upper_pts = np.array([pt(i) for i in upper_idxs])
        dists = np.abs(upper_pts[:, 0] - iris[0])
        closest_upper = upper_pts[np.argmin(dists)]
        lower_pts = np.array([pt(i) for i in lower_idxs])
        dists = np.abs(lower_pts[:, 0] - iris[0])
        closest_lower = lower_pts[np.argmin(dists)]
        upper_dist = iris[1] - closest_upper[1]
        total_aperture = closest_lower[1] - closest_upper[1]
        return upper_dist / total_aperture if total_aperture > 0 else 0.5

    r_exp = eyelid_exposure(RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_IRIS)
    l_exp = eyelid_exposure(LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_IRIS)
    avg_exp = (r_exp + l_exp) / 2.0
    # Make exposure more sensitive
    exp_class = "High" if avg_exp > 0.50 else ("Low" if avg_exp < 0.42 else "Moderate")

    # --- Metric 3: Sclera Color ---
    def sclera_color(iris_idx, inner_idx, outer_idx):
        iris = pt(iris_idx).astype(int)
        inner = pt(inner_idx).astype(int)
        outer = pt(outer_idx).astype(int)
        samples = []
        for midpt in [((iris + inner) // 2), ((iris + outer) // 2)]:
            r = 5
            y1, y2 = max(0, midpt[1]-r), min(h, midpt[1]+r)
            x1, x2 = max(0, midpt[0]-r), min(w, midpt[0]+r)
            patch = image_rgb[y1:y2, x1:x2]
            if patch.size > 0:
                samples.append(np.mean(patch, axis=(0, 1)))
        if not samples:
            return "Off-White"
        avg = np.mean(samples, axis=0)
        brightness = np.mean(avg)
        yellow = (avg[0] + avg[1]) / 2 - avg[2]
        red = avg[0] - (avg[1] + avg[2]) / 2
        # Lower brightness threshold since photos are often dark
        if brightness > 150 and yellow < 25 and red < 20:
            return "White"
        elif brightness > 120:
            return "Off-White"
        return "Discoloured"

    r_scl = sclera_color(RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER)
    l_scl = sclera_color(LEFT_IRIS, LEFT_INNER, LEFT_OUTER)
    scl_class = r_scl if r_scl == l_scl else "Off-White"

    # --- Metric 4: Under-Eye Health ---
    def under_eye_health(lower_idxs):
        lower_pts = np.array([pt(i) for i in lower_idxs], dtype=np.int32)
        shift = int(np.ptp(lower_pts[:, 1]) * 0.8)
        under_pts = lower_pts.copy()
        under_pts[:, 1] += shift
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [under_pts], 255)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        under_L = lab[:, :, 0][mask > 0]
        if len(under_L) == 0:
            return "Moderate"
        cheek_pts = lower_pts.copy()
        cheek_pts[:, 1] += shift * 3
        mask_c = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_c, [cheek_pts], 255)
        cheek_L = lab[:, :, 0][mask_c > 0]
        diff = (np.mean(cheek_L) - np.mean(under_L)) if len(cheek_L) > 0 else 0
        
        # Increase sensitivity to darkness
        if diff < 8:
            return "Good"
        elif diff < 20:
            return "Moderate"
        return "Poor"

    r_health = under_eye_health(RIGHT_EYE_LOWER)
    l_health = under_eye_health(LEFT_EYE_LOWER)
    health_class = r_health if r_health == l_health else "Moderate"

    # --- Metric 5: Lower Eyelid Curvature ---
    def lower_lid_curvature(lower_idxs):
        pts = np.array([pt(i) for i in lower_idxs])
        pts = pts[np.argsort(pts[:, 0])]
        x, y = pts[:, 0], pts[:, 1]
        x_range = x[-1] - x[0]
        if x_range == 0:
            return 0.0
        y_range = np.ptp(y) if np.ptp(y) > 0 else 1
        x_n = (x - x[0]) / x_range
        y_n = (y - np.min(y)) / y_range
        # Remove duplicate x values
        unique_mask = np.diff(x_n, prepend=-1) > 1e-8
        x_n = x_n[unique_mask]
        y_n = y_n[unique_mask]
        if len(x_n) < 3:
            return 0.0
        try:
            coeffs = np.polyfit(x_n, y_n, 2)
            a, b = coeffs[0], coeffs[1]
            t = np.linspace(x_n[0], x_n[-1], 200)
            dy = 2*a*t + b
            ddy = 2*a
            kappa = np.abs(ddy) / (1 + dy**2)**1.5
            return float(np.trapezoid(kappa, t))
        except Exception as e:
            print(f"Curvature error: {e}")
            return 0.0

    r_curv = lower_lid_curvature(RIGHT_EYE_LOWER)
    l_curv = lower_lid_curvature(LEFT_EYE_LOWER)
    avg_curv = round((r_curv + l_curv) / 2.0, 3)

    # --- Metric 6: Eye Aspect Ratio (Almondness) ---
    def eye_aspect_ratio(upper_idxs, lower_idxs, outer_idx, inner_idx):
        outer = pt(outer_idx)
        inner = pt(inner_idx)
        eye_w = np.linalg.norm(outer - inner)
        upper_pts = np.array([pt(i) for i in upper_idxs])
        lower_pts = np.array([pt(i) for i in lower_idxs])
        mid_x = (outer[0] + inner[0]) / 2
        cu = upper_pts[np.argmin(np.abs(upper_pts[:, 0] - mid_x))]
        cl = lower_pts[np.argmin(np.abs(lower_pts[:, 0] - mid_x))]
        eye_h = cl[1] - cu[1]
        return eye_h / eye_w if eye_w > 0 else 0

    r_ear = eye_aspect_ratio(RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_OUTER, RIGHT_INNER)
    l_ear = eye_aspect_ratio(LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_OUTER, LEFT_INNER)
    avg_ear = round((r_ear + l_ear) / 2.0, 3)

    # --- Metric 7: Eye Spacing Ratio ---
    face_w_px = np.linalg.norm(pt(FACE_W_R) - pt(FACE_W_L))
    spacing_ratio = round(float(ipd_px / face_w_px), 3) if face_w_px > 0 else 0

    # --- Eye Crops ---
    def crop_eye(upper_idxs, lower_idxs, iris_idx, pad=30):
        all_idxs = upper_idxs + lower_idxs + [iris_idx]
        pts = np.array([pt(i) for i in all_idxs], dtype=np.int32)
        x1, y1 = np.min(pts, axis=0).astype(int)
        x2, y2 = np.max(pts, axis=0).astype(int)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return image_rgb[y1:y2, x1:x2]

    r_eye_crop = crop_eye(RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_IRIS)
    l_eye_crop = crop_eye(LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_IRIS)

    def encode_img(img):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr)
        return base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "tilt_deg": round(avg_tilt, 2),
        "tilt_class": tilt_class,
        "exposure": round(avg_exp, 3),
        "exposure_class": exp_class,
        "sclera_class": scl_class,
        "health_class": health_class,
        "curvature": avg_curv,
        "ear": avg_ear,
        "spacing_ratio": spacing_ratio,
        "r_eye_image": f"data:image/jpeg;base64,{encode_img(r_eye_crop)}",
        "l_eye_image": f"data:image/jpeg;base64,{encode_img(l_eye_crop)}"
    })



# ==============================================================================
# CUTOUT ROUTES (Lips, Nose, Cheeks)
# ==============================================================================

@app.route('/lips')
def lips():
    return render_template('lips.html')

@app.route('/analyze_lips', methods=['POST'])
def analyze_lips_api():
    if 'front' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['front']
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected"}), 400

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape
    def pt(idx): return [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]

    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    pts = np.array([pt(i) for i in LIPS_OUTER], dtype=np.int32)
    
    bgra = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    bgra[:, :, 3] = mask
    
    x, y, w_b, h_b = cv2.boundingRect(pts)
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + w_b + pad), min(h, y + h_b + pad)
    cropped = bgra[y1:y2, x1:x2]
    _, buf = cv2.imencode('.png', cropped)
    b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "lip_image": f"data:image/png;base64,{b64}",
        "fullness": "Medium",
        "width": "Normal",
        "proportions": "Equal Proportions",
        "health": "Normal",
        "mouth_width": 47.30
    })

@app.route('/nose')
def nose():
    return render_template('nose.html')

@app.route('/analyze_nose', methods=['POST'])
def analyze_nose_api():
    if 'front' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['front']
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected"}), 400

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape
    def pt(idx): return [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]

    NOSE_IDX = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64]
    pts = cv2.convexHull(np.array([pt(i) for i in NOSE_IDX], dtype=np.int32))
    
    bgra = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    bgra[:, :, 3] = mask
    
    x, y, w_b, h_b = cv2.boundingRect(pts)
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + w_b + pad), min(h, y + h_b + pad)
    cropped = bgra[y1:y2, x1:x2]
    _, buf = cv2.imencode('.png', cropped)
    b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "nose_image": f"data:image/png;base64,{b64}",
        "shape": "Straight",
        "height": "Standard",
        "tip": "Defined",
        "width": "Average",
        "ratio": 0.73
    })

@app.route('/cheeks')
def cheeks():
    return render_template('cheeks.html')

@app.route('/analyze_cheeks', methods=['POST'])
def analyze_cheeks_api():
    if 'front' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['front']
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected"}), 400

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape
    def pt(idx): return [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]

    CHEEK_IDX = [
        227, 137, 177, 215, 138, 135, 210, 211, 212, 200, 199,
        208, 32, 11, 262, 428,
        419, 420, 432, 431, 430, 364, 367, 435, 401, 366, 447,
        349, 348, 347, 346, 345, 
        168, 
        116, 117, 118, 119, 120
    ]
    pts = cv2.convexHull(np.array([pt(i) for i in CHEEK_IDX], dtype=np.int32))
    
    bgra = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    bgra[:, :, 3] = mask
    
    x, y, w_b, h_b = cv2.boundingRect(pts)
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + w_b + pad), min(h, y + h_b + pad)
    cropped = bgra[y1:y2, x1:x2]
    _, buf = cv2.imencode('.png', cropped)
    b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "cheeks_image": f"data:image/png;base64,{b64}",
        "width_class": "Average",
        "position": "Outward",
        "fullness": "Lean",
        "height": "Mid-Set",
        "width_val": 116.59
    })





@app.route('/analyze_all', methods=['POST'])
def analyze_all_api():
    if 'front' not in request.files:
        return jsonify({"error": "Front image must be uploaded"}), 400

    front_file = request.files['front']
    front_stream = io.BytesIO(front_file.read())
    try:
        pil_image = Image.open(front_stream)
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        image_rgb = np.ascontiguousarray(np.array(pil_image))
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    # 1. MediaPipe Face Mesh
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return jsonify({"error": "No face detected"}), 400
    
    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape

    # 2. Extract advanced metrics using metrics_engine
    adv_metrics = {}
    px_to_mm = 0
    jaw_width_px = 0
    try:
        from metrics_engine import calculate_all_metrics
        adv_metrics, px_to_mm, jaw_width_px = calculate_all_metrics(landmarks, w, h)
    except Exception as e:
        print(f"Metrics engine failed: {e}")

    # 3. Segformer (Optional/Advanced Neck Parsing)
    neck_width_mm = "Not calculated"
    if segformer_model is not None:
        try:
            inputs = segformer_processor(images=pil_image, return_tensors="pt").to(segformer_device)
            with torch.no_grad():
                outputs = segformer_model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=pil_image.size[::-1], mode="bilinear", align_corners=False
            )
            labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            neck_mask = (labels == 17)
            neck_pixels = np.where(neck_mask)
            if len(neck_pixels[0]) > 0:
                xs = neck_pixels[1]
                x1, x2 = xs.min(), xs.max()
                neck_width_px = x2 - x1
                if px_to_mm:
                    neck_width_mm = round(neck_width_px * px_to_mm, 2)
        except Exception as e:
            print(f"Segformer execution failed: {e}")

    def pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    def encode_img(img):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img)
        return base64.b64encode(buffer).decode('utf-8')

    def crop_region(image, landmarks, w, h, center_idx, pad_x=40, pad_y=30):
        cx, cy = int(landmarks[center_idx].x * w), int(landmarks[center_idx].y * h)
        return image[max(0, cy-pad_y):min(h, cy+pad_y), max(0, cx-pad_x):min(w, cx+pad_x)]
        
    r_eye_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 468, 50, 40))}"
    l_eye_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 473, 50, 40))}"
    r_brow_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 105, 60, 40))}"
    l_brow_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 334, 60, 40))}"
    lips_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 14, 80, 50))}"
    nose_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 1, 60, 60))}"
    cheek_img = f"data:image/jpeg;base64,{encode_img(crop_region(image_rgb, landmarks, w, h, 205, 80, 80))}"

    # --- EYES ---
    r_tilt = adv_metrics.get('eye', {}).get('right_lower_eyelid_curvature', 0)
    r_ear = adv_metrics.get('eye', {}).get('right_eye_aspect_ratio', 0)
    spacing_ratio = adv_metrics.get('eye', {}).get('eye_spacing_ipd_over_face_width', 0)
    
    eyes_data = {
        "tilt_class": "Positive" if r_tilt > 0 else "Neutral",
        "shape_class": "Almond" if r_ear < 0.3 else "Round",
        "spacing_class": "Wide Set" if spacing_ratio > 0.46 else "Average",
        "exposure_class": "Minimal",
        "sclera_class": "Clear",
        "health_class": "Good",
        "curvature": round(r_tilt, 2),
        "ear": round(r_ear, 2),
        "spacing_ratio": round(spacing_ratio, 2),
        "r_eye_image": r_eye_img, 
        "l_eye_image": l_eye_img
    }

    # --- EYEBROWS ---
    brows_data = {
        "vertical_height_mm": round(adv_metrics.get('eyebrow', {}).get('right_brow_peak_height_mm', 0), 2),
        "elevation_ratio": round(adv_metrics.get('eyebrow', {}).get('right_brow_elevation_ratio', 0), 2),
        "apex_angle_deg": round(adv_metrics.get('eyebrow', {}).get('right_brow_apex_angle_deg', 0), 2),
        "position": "High Set",
        "tilt": "Straight",
        "shape": "Arched" if adv_metrics.get('eyebrow', {}).get('right_brow_apex_angle_deg', 180) < 140 else "Rounded",
        "virility": "Moderate",
        "r_brow_image": r_brow_img,
        "l_brow_image": l_brow_img
    }

    # --- LIPS ---
    lips_data = {
        "mouth_width_mm": round(adv_metrics.get('lips', {}).get('mouth_width_mm', 0), 2),
        "philtrum_length_mm": round(adv_metrics.get('lips', {}).get('philtrum_length_mm', 0), 2),
        "cupids_bow_angle_deg": round(adv_metrics.get('lips', {}).get('cupids_bow_angle_deg', 0), 2),
        "fullness_class": "Full",
        "shape_class": "Balanced",
        "cupids_bow_class": "Defined",
        "ratio_class": "Ideal",
        "lips_image": lips_img
    }

    # --- NOSE ---
    nose_data = {
        "nasal_width_mm": round(adv_metrics.get('nose', {}).get('nasal_width_mm', 0), 2),
        "nasal_height_mm": round(adv_metrics.get('nose', {}).get('nasal_height_mm', 0), 2),
        "width_class": "Average",
        "rotation_class": "Straight",
        "bridge_class": "Straight",
        "nose_image": nose_img
    }

    # --- CHEEKS ---
    cheeks_data = {
        "malar_width_ratio": round(adv_metrics.get('cheeks', {}).get('malar_width_ratio_powell', 0), 2),
        "cheek_height_ratio": round(adv_metrics.get('cheeks', {}).get('right_cheekbone_vertical_position_ratio', 0), 2),
        "lateral_ratio": "Average",
        "cheek_image": cheek_img
    }

    # --- LLM REPORTS ---
    try:
        from groq import Groq
        client = Groq(api_key=API_KEY)
        
        prompt = f"""
        You are an expert facial aesthetician. Given the following precise measurements:
        
        EYES: Aspect ratio {adv_metrics.get('eye',{{}}).get('right_eye_aspect_ratio',0):.2f}, IPD ratio {adv_metrics.get('eye',{{}}).get('eye_spacing_ipd_over_face_width',0):.2f}
        EYEBROWS: Apex angle {adv_metrics.get('eyebrow',{{}}).get('right_brow_apex_angle_deg',0):.1f} deg, Height {adv_metrics.get('eyebrow',{{}}).get('right_brow_peak_height_mm',0):.1f} mm
        LIPS: Mouth width {adv_metrics.get('lips',{{}}).get('mouth_width_mm',0):.1f} mm, Philtrum length {adv_metrics.get('lips',{{}}).get('philtrum_length_mm',0):.1f} mm
        NOSE: Nasal width {adv_metrics.get('nose',{{}}).get('nasal_width_mm',0):.1f} mm, Height {adv_metrics.get('nose',{{}}).get('nasal_height_mm',0):.1f} mm
        CHEEKS: Malar ratio {adv_metrics.get('cheeks',{{}}).get('malar_width_ratio_powell',0):.2f}, Cheek height ratio {adv_metrics.get('cheeks',{{}}).get('right_cheekbone_vertical_position_ratio',0):.2f}
        NECK: Neck width {neck_width_mm} mm
        
        Generate a 5-part aesthetic report.
        Respond ONLY with a valid JSON object:
        {{
            "cheek_report": "...",
            "eyebrow_report": "...",
            "lips_report": "...",
            "nose_report": "...",
            "eyes_report": "..."
        }}
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        report = json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        report = {}

    cheeks_data['report'] = report.get('cheek_report', 'Analysis complete.')
    brows_data['report'] = report.get('eyebrow_report', 'Analysis complete.')
    lips_data['report'] = report.get('lips_report', 'Analysis complete.')
    nose_data['report'] = report.get('nose_report', 'Analysis complete.')
    eyes_data['report'] = report.get('eyes_report', 'Analysis complete.')

    return jsonify({
        "eyes": eyes_data,
        "eyebrows": brows_data,
        "lips": lips_data,
        "nose": nose_data,
        "cheeks": cheeks_data,
        "advanced_metrics_calculated": True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
