"""
Face Parsing & Metrics Web App
Standalone Flask app based on the face-parsing-with-metrics-structured notebook.
Runs on port 5001.
"""
import os, io, json, base64, math
import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify

# --- Torch + SegFormer ---
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- MediaPipe ---
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ====== GLOBALS ======
app = Flask(__name__)

LABEL_MAP = {
    0: "background", 1: "skin", 2: "nose", 3: "eye_g",
    4: "left_eye", 5: "right_eye", 6: "left_eyebrow", 7: "right_eyebrow",
    8: "left_ear", 9: "right_ear", 10: "mouth", 11: "upper_lip",
    12: "lower_lip", 13: "hair", 14: "hat", 15: "earring",
    16: "necklace", 17: "neck", 18: "cloth",
}

# Nice colors for each label (RGB)
LABEL_COLORS = {
    0: (0,0,0), 1: (204,204,255), 2: (255,204,153), 3: (153,204,255),
    4: (102,178,255), 5: (102,178,255), 6: (255,178,102), 7: (255,178,102),
    8: (178,102,255), 9: (178,102,255), 10: (255,102,178), 11: (255,102,102),
    12: (255,153,153), 13: (102,51,0), 14: (255,255,102), 15: (255,204,0),
    16: (204,153,0), 17: (153,255,204), 18: (102,102,153),
}

# ====== INIT MODELS ======
print("Loading SegFormer face parsing model...")
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
device = "cuda" if torch.cuda.is_available() else "cpu"
segformer_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing", token=os.environ["HF_TOKEN"])
segformer_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing", token=os.environ["HF_TOKEN"])
segformer_model.to(device)
segformer_model.eval()
print(f"  SegFormer loaded on {device}")

print("Loading MediaPipe Face Landmarker...")
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    import urllib.request
    print("  Downloading face_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)
print("  MediaPipe loaded")
print("All models ready!\n")


# ====== HELPER FUNCTIONS ======
def encode_cv2(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

def encode_rgb(img_rgb):
    return encode_cv2(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def run_segmentation(pil_image):
    inputs = segformer_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=pil_image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()
    return labels

def make_segmentation_overlay(img_rgb, labels):
    h, w = labels.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in LABEL_COLORS.items():
        overlay[labels == idx] = color
    blended = cv2.addWeighted(img_rgb, 0.4, overlay, 0.6, 0)
    return blended

def crop_part(img_rgb, labels, label_id, padding=15):
    mask = (labels == label_id)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = img_rgb.shape[:2]
    x1, x2 = max(0, xs.min() - padding), min(w, xs.max() + padding)
    y1, y2 = max(0, ys.min() - padding), min(h, ys.max() + padding)
    part = np.ones_like(img_rgb) * 255
    part[mask] = img_rgb[mask]
    cropped = part[y1:y2, x1:x2]
    return cropped

def dist_between(landmarks, i, j, w, h):
    p1 = np.array([landmarks[i].x * w, landmarks[i].y * h])
    p2 = np.array([landmarks[j].x * w, landmarks[j].y * h])
    return float(np.linalg.norm(p2 - p1))

def compute_metrics(landmarks, w, h):
    pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])

    def lm(i): return pts[i].astype(float)
    def dist(i, j): return float(np.linalg.norm(lm(j) - lm(i)))
    def angle(vertex, a, b):
        v, pa, pb = lm(vertex), lm(a), lm(b)
        va, vb = pa - v, pb - v
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        return round(float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))), 1)
    def curv(s, m, e):
        p1, p2, p3 = lm(s), lm(m), lm(e)
        chord = p3 - p1; cl = np.linalg.norm(chord)
        if cl < 1e-6: return 0.0
        t = np.dot(p2 - p1, chord) / (cl**2)
        proj = p1 + t * chord
        return round(float(np.linalg.norm(p2 - proj) / cl), 3)
    def ear(p1,p2,p3,p4,p5,p6):
        pts_l = [lm(i) for i in (p1,p2,p3,p4,p5,p6)]
        vert = np.linalg.norm(pts_l[1]-pts_l[5]) + np.linalg.norm(pts_l[2]-pts_l[4])
        horiz = np.linalg.norm(pts_l[0]-pts_l[3])
        return round(float(vert / (2*horiz+1e-9)), 3)

    FT, GL, NT, NB, SN, ME = 10, 9, 4, 168, 2, 152
    RZ, LZ, RT, LT, RG, LG = 234, 454, 127, 356, 172, 397

    fw = dist(RZ, LZ); fh = dist(FT, ME); ipd = dist(33, 263)
    mm = 63.5 / (ipd + 1e-9)

    metrics = {}

    metrics["Eyebrows"] = {
        "Right Brow Peak Height": f"{abs(lm(105)[1]-lm(159)[1])*mm:.1f} mm",
        "Left Brow Peak Height": f"{abs(lm(334)[1]-lm(386)[1])*mm:.1f} mm",
        "Right Brow Elevation Ratio": f"{abs(lm(105)[1]-lm(159)[1])/(ipd+1e-9):.3f}",
        "Left Brow Elevation Ratio": f"{abs(lm(334)[1]-lm(386)[1])/(ipd+1e-9):.3f}",
        "Right Brow Apex Angle": f"{angle(105, 55, 70)}°",
        "Left Brow Apex Angle": f"{angle(334, 285, 300)}°",
    }

    metrics["Eyes"] = {
        "Right Eye Aspect Ratio": f"{ear(33,160,158,133,153,144)}",
        "Left Eye Aspect Ratio": f"{ear(362,385,387,263,373,380)}",
        "IPD / Face Width Ratio": f"{ipd/(fw+1e-9):.3f}",
        "Right Lower Eyelid Curvature": f"{curv(33, 145, 133)}",
        "Left Lower Eyelid Curvature": f"{curv(362, 374, 263)}",
    }

    nw = dist(129,358); nh = dist(NB,SN); ic = dist(133,362)
    metrics["Nose"] = {
        "Nasal Width": f"{nw*mm:.1f} mm",
        "Nasal Height": f"{nh*mm:.1f} mm",
        "Nasal Aspect Ratio (W/H)": f"{nw/(nh+1e-9):.3f}",
        "Naso-Canthal Ratio": f"{nw/(ic+1e-9):.3f}",
        "Pyramidal Width": f"{dist(236,456)*mm:.1f} mm",
    }

    mw = dist(61,291)
    metrics["Lips"] = {
        "Mouth Width": f"{mw*mm:.1f} mm",
        "Philtrum Length": f"{dist(SN,0)*mm:.1f} mm",
        "Cupid's Bow Angle": f"{angle(0, 37, 267)}°",
    }

    metrics["Cheeks"] = {
        "Facial Width": f"{fw*mm:.1f} mm",
        "Malar Width Ratio (Powell)": f"{fw/(fh+1e-9):.3f}",
        "R Cheekbone Vertical Pos": f"{(lm(RZ)[1]-lm(FT)[1])/(fh+1e-9):.3f}",
        "L Cheekbone Vertical Pos": f"{(lm(LZ)[1]-lm(FT)[1])/(fh+1e-9):.3f}",
    }

    jw = dist(RG, LG)
    jaw_vec = lm(ME) - lm(RG)
    jaw_incl = round(float(np.degrees(np.arccos(np.clip(
        np.dot(jaw_vec, [1,0]) / (np.linalg.norm(jaw_vec) + 1e-9), -1, 1
    )))), 1)
    metrics["Jaw"] = {
        "Jaw Rise": f"{abs(lm(RG)[1]-lm(ME)[1])*mm:.1f} mm",
        "Jaw Width": f"{jw*mm:.1f} mm",
        "R Jaw Inclination": f"{jaw_incl}°",
        "Face Width": f"{fw*mm:.1f} mm",
    }

    gl_nt = lm(NT) - lm(GL)
    gl_nt_dot = np.dot(gl_nt, gl_nt) + 1e-9
    proj = lm(GL) + np.dot(lm(ME) - lm(GL), gl_nt) / gl_nt_dot * gl_nt
    chin_dev = round(float(np.linalg.norm(lm(ME) - proj)) * mm, 1)
    metrics["Chin"] = {
        "Chin Width": f"{dist(214,434)*mm:.1f} mm",
        "Chin Vertical Height": f"{dist(17,ME)*mm:.1f} mm",
        "Chin Midline Deviation": f"{chin_dev} mm",
    }

    metrics["Hair / Forehead"] = {
        "Forehead Width": f"{dist(RT,LT)*mm:.1f} mm",
        "Forehead Height (Approx)": f"{dist(FT,GL)*mm:.1f} mm",
    }

    metrics["Smile"] = {
        "Upper Arc Curvature": f"{curv(61, 0, 291)}",
        "Lower Arc Curvature": f"{curv(61, 17, 291)}",
        "Smile Width": f"{mw*mm:.1f} mm",
    }

    metrics["Neck"] = {}

    summary = {
        "Face Width (px)": f"{fw:.1f}",
        "Face Height (px)": f"{fh:.1f}",
        "IPD (px)": f"{ipd:.1f}",
        "Scale Factor": f"{mm:.4f} mm/px",
    }

    return metrics, summary, mm

@app.route('/')
def index():
    return render_template('face_parsing.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    stream = io.BytesIO(file.read())
    try:
        pil_image = Image.open(stream).convert('RGB')
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    img_rgb = np.array(pil_image)
    h, w = img_rgb.shape[:2]

    labels = run_segmentation(pil_image)
    overlay = make_segmentation_overlay(img_rgb, labels)
    overlay_b64 = encode_rgb(overlay)

    parts = {}
    for idx, name in LABEL_MAP.items():
        if idx == 0:
            continue
        crop = crop_part(img_rgb, labels, idx)
        if crop is not None:
            parts[name] = encode_rgb(crop)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
    detection_result = landmarker.detect(mp_image)

    metrics = {}
    summary = {}
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        class LandmarkWrapper:
            def __init__(self, lms):
                self.landmark = lms
        wrapper = LandmarkWrapper(landmarks)
        metrics, summary, mm_per_px = compute_metrics(wrapper, w, h)

        neck_mask = (labels == 17)
        neck_px = np.where(neck_mask)
        if len(neck_px[0]) > 0:
            nx1, nx2 = neck_px[1].min(), neck_px[1].max()
            nw_px = nx2 - nx1
            metrics["Neck"] = {
                "Neck Width": f"{nw_px * mm_per_px:.1f} mm",
                "Neck/Jaw Ratio": f"{nw_px / (dist_between(landmarks, 172, 397, w, h) + 1e-9):.3f}",
            }

        landmark_img = img_rgb.copy()
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(landmark_img, (cx, cy), 1, (0, 255, 0), -1)
        landmarks_b64 = encode_rgb(landmark_img)
    else:
        landmarks_b64 = encode_rgb(img_rgb)
        metrics = {"Error": {"message": "No face landmarks detected"}}

    original_b64 = encode_rgb(img_rgb)
    unique_labels = [LABEL_MAP.get(int(l), f"unknown_{l}") for l in np.unique(labels) if l != 0]

    return jsonify({
        "original": original_b64,
        "overlay": overlay_b64,
        "landmarks": landmarks_b64,
        "parts": parts,
        "metrics": metrics,
        "summary": summary,
        "detected_regions": unique_labels,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
