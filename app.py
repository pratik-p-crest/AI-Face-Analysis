import os
import io
import json
import base64
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
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'front' not in request.files or 'left' not in request.files or 'right' not in request.files:
        return jsonify({"error": "All 3 images (front, left, right) must be uploaded"}), 400
        
    front_file = request.files['front']
    left_file = request.files['left']
    right_file = request.files['right']
    
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

    face_landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape

    # 1. CHEEK ANALYSIS
    r_cheek = get_landmark_pt(face_landmarks, 234, w, h)
    l_cheek = get_landmark_pt(face_landmarks, 454, w, h)
    r_jaw = get_landmark_pt(face_landmarks, 132, w, h)
    l_jaw = get_landmark_pt(face_landmarks, 361, w, h)
    chin = get_landmark_pt(face_landmarks, 152, w, h)
    top_head = get_landmark_pt(face_landmarks, 10, w, h)
    r_eye = get_landmark_pt(face_landmarks, 33, w, h)
    l_eye = get_landmark_pt(face_landmarks, 263, w, h)

    cheek_width = np.linalg.norm(r_cheek - l_cheek)
    jaw_width = np.linalg.norm(r_jaw - l_jaw)
    lateral_ratio = cheek_width / jaw_width

    face_height = chin[1] - top_head[1]
    avg_eye_y = (r_eye[1] + l_eye[1]) / 2.0
    avg_cheek_y = (r_cheek[1] + l_cheek[1]) / 2.0
    cheek_height_ratio = (avg_cheek_y - avg_eye_y) / face_height

    # 2. EYEBROW ANALYSIS
    # Eyebrow landmarks: Right (70, 63, 105, 66, 107), Left (336, 296, 334, 293, 300)
    # Distance between brows: 107 and 336
    r_inner = get_landmark_pt(face_landmarks, 107, w, h)
    l_inner = get_landmark_pt(face_landmarks, 336, w, h)
    inter_brow_dist = np.linalg.norm(r_inner - l_inner)
    eye_dist = np.linalg.norm(r_eye - l_eye)
    brow_distance_ratio = inter_brow_dist / eye_dist
    
    # Arch height: Right brow
    r_arch = get_landmark_pt(face_landmarks, 105, w, h)
    r_outer = get_landmark_pt(face_landmarks, 70, w, h)
    brow_width = np.linalg.norm(r_outer - r_inner)
    arch_height = r_inner[1] - r_arch[1] # Negative Y is up
    brow_arch_ratio = arch_height / brow_width

    # 3. EAR ANALYSIS
    left_ear_meas, right_ear_meas = {}, {}
    l_ear_crop, r_ear_crop = None, None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        left_path = os.path.join(tmpdir, "left.jpg")
        right_path = os.path.join(tmpdir, "right.jpg")
        left_file.seek(0)
        right_file.seek(0)
        Image.open(left_file).save(left_path)
        Image.open(right_file).save(right_path)
        
        try:
            left_result = run_roboflow_ear_workflow(left_path)
            res = draw_ear_calipers(left_path, left_result)
            if res: l_ear_crop, left_ear_meas = res
        except Exception as e:
            print("Left Ear error:", e)
            
        try:
            right_result = run_roboflow_ear_workflow(right_path)
            res = draw_ear_calipers(right_path, right_result)
            if res: r_ear_crop, right_ear_meas = res
        except Exception as e:
            print("Right Ear error:", e)

    # 4. LLM GENERATION
    if not API_KEY or API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return jsonify({"error": "Groq API Key not configured in .env"}), 500

    prompt = f"""
    You are an expert facial aesthetician. I am analyzing a client's face based on geometric AI data.

    CHEEK DATA:
    - Lateral Projection Ratio (Cheek/Jaw): {lateral_ratio:.3f} (>1.15 is highly projecting).
    - Cheek Height Ratio: {cheek_height_ratio:.3f} (<0.12 means high cheekbones).

    EYEBROW DATA:
    - Brow Arch Ratio: {brow_arch_ratio:.3f} (Higher means more arched/feminine, lower means flat/masculine).
    - Inter-brow Distance Ratio: {brow_distance_ratio:.3f} (Lower means close-set).
    
    EAR DATA:
    - Left Ear Upper Width: {left_ear_meas.get('ear_width_top_px', 0):.1f}px, Height: {left_ear_meas.get('ear_height_px', 0):.1f}px
    - Right Ear Upper Width: {right_ear_meas.get('ear_width_top_px', 0):.1f}px, Height: {right_ear_meas.get('ear_height_px', 0):.1f}px

    TASK:
    Generate a 3-part aesthetic report.
    1. 'cheek_report': A short paragraph on their cheek structure and aesthetic suggestions.
    2. 'eyebrow_report': A short paragraph on their eyebrow shape and suggestions (e.g. arched vs flat).
    3. 'ear_report': A short paragraph comparing left/right ears or their general prominence based on width/height ratios.
    
    Respond ONLY with a valid JSON object:
    {{
        "cheek_report": "...",
        "eyebrow_report": "...",
        "ear_report": "..."
    }}
    """
    
    try:
        client = Groq(api_key=API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an API that outputs ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        report = json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        return jsonify({"error": f"Groq API Error: {str(e)}"}), 500

    # 5. DRAW ANNOTATIONS
    annotated_image = image_rgb.copy()
    line_color = (255, 255, 255)
    thickness = 1

    def get_pt_tuple(idx):
        return (int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h))

    # Cheek Vectors
    r_zygion = get_pt_tuple(234)
    r_nose = get_pt_tuple(129)
    r_mouth = get_pt_tuple(61)
    l_zygion = get_pt_tuple(454)
    l_nose = get_pt_tuple(358)
    l_mouth = get_pt_tuple(291)
    chin_tuple = get_pt_tuple(152)

    cv2.line(annotated_image, r_zygion, r_nose, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, r_zygion, r_mouth, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, r_zygion, chin_tuple, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, l_zygion, l_nose, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, l_zygion, l_mouth, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, l_zygion, chin_tuple, line_color, thickness, cv2.LINE_AA)
    cv2.line(annotated_image, r_zygion, l_zygion, line_color, thickness, cv2.LINE_AA)
    cv2.drawMarker(annotated_image, r_zygion, line_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2, line_type=cv2.LINE_AA)
    cv2.drawMarker(annotated_image, l_zygion, line_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2, line_type=cv2.LINE_AA)

    # Eyebrow Cropping
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

    r_eyebrow_crop = crop_eyebrow(image_rgb, face_landmarks, right_eyebrow_idxs, w, h)
    l_eyebrow_crop = crop_eyebrow(image_rgb, face_landmarks, left_eyebrow_idxs, w, h)
    
    # Full Image Eyebrow Mapping (Red Dots)
    eyebrow_full_annotated = image_rgb.copy()
    for idx in right_eyebrow_idxs + left_eyebrow_idxs:
        pt = (int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h))
        cv2.circle(eyebrow_full_annotated, pt, 3, (255, 0, 0), -1)

    # Encode Images
    def encode_img(img):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img)
        return base64.b64encode(buffer).decode('utf-8')

    base64_cheek = encode_img(annotated_image)
    base64_r_brow = encode_img(r_eyebrow_crop)
    base64_l_brow = encode_img(l_eyebrow_crop)
    base64_eyebrow_full = encode_img(eyebrow_full_annotated)
    
    def encode_pil(img):
        if not img: return ""
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        "cheek_image": f"data:image/jpeg;base64,{base64_cheek}",
        "eyebrow_full_image": f"data:image/jpeg;base64,{base64_eyebrow_full}",
        "r_brow_image": f"data:image/jpeg;base64,{base64_r_brow}",
        "l_brow_image": f"data:image/jpeg;base64,{base64_l_brow}",
        "l_ear_image": f"data:image/jpeg;base64,{encode_pil(l_ear_crop)}" if l_ear_crop else "",
        "r_ear_image": f"data:image/jpeg;base64,{encode_pil(r_ear_crop)}" if r_ear_crop else "",
        "cheek_report": report.get("cheek_report", ""),
        "eyebrow_report": report.get("eyebrow_report", ""),
        "ear_report": report.get("ear_report", "")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
