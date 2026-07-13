# 1. Import the library
from inference_sdk import InferenceHTTPClient
 
# 2. Connect to your workspace
client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="R1zODk2Ja9JHBabfRASg"
)
 
# 3. Run your workflow on an image
result_2 = client.run_workflow(
  workspace_name="fabiki4429-acoxs-com",
  workflow_id="general-segmentation-api-2",
  images={
    "image": "images.jpg"  # Path to your image file
  },
  parameters={
    "classes": "ear, lower_ear, upper_ear"
  },
  use_cache=True  # cache workflow definition for 15 minutes
)
 
# 4. Get your results
print(result)
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# Read image
image = cv2.imread("left-profile.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
prediction = result[0]['predictions']['predictions'][0]
 
# Polygon points
points = np.array([[p["x"], p["y"]] for p in prediction["points"]], dtype=np.int32)
 
# Bounding box from polygon
x_min = points[:,0].min()
x_max = points[:,0].max()
y_min = points[:,1].min()
y_max = points[:,1].max()
 
# -------------------------
# Draw polygon (optional)
# -------------------------
cv2.polylines(image, [points], True, (255,0,0), 2)
 
# =========================
# Vertical measurement line
# =========================
 
offset = 20   # distance from ear
 
# Main vertical line
cv2.line(image,
         (x_min-offset, y_min),
         (x_min-offset, y_max),
         (255,255,255),2)
 
# Top cap
cv2.line(image,
         (x_min-offset-5, y_min),
         (x_min-offset+5, y_min),
         (255,255,255),2)
 
# Bottom cap
cv2.line(image,
         (x_min-offset-5, y_max),
         (x_min-offset+5, y_max),
         (255,255,255),2)
 
# =========================
# Horizontal measurement
# =========================
 
y_mid = int((y_min+y_max)/2)
 
cv2.line(image,
         (x_min, y_mid),
         (x_max, y_mid),
         (255,255,255),2)
 
# Left cap
cv2.line(image,
         (x_min, y_mid-5),
         (x_min, y_mid+5),
         (255,255,255),2)
 
# Right cap
cv2.line(image,
         (x_max, y_mid-5),
         (x_max, y_mid+5),
         (255,255,255),2)
 
plt.figure(figsize=(6,8))
plt.imshow(image)
plt.axis("off")
plt.show()
 
import json
import math
import sys
from PIL import Image, ImageDraw
 
 
def _extract_ear_points(api_result, class_name="ear"):
    """
    Pulls the (x, y) polygon points for the requested class out of a
    run_workflow() result. Works whether api_result is the raw list
    returned by the SDK, or a single prediction dict, or a JSON string.
    """
    if isinstance(api_result, str):
        api_result = json.loads(api_result)
 
    # run_workflow() returns a list with one dict per input image
    if isinstance(api_result, list):
        api_result = api_result[0]
 
    predictions_block = api_result["predictions"]
    img_meta = predictions_block["image"]           # {"width":.., "height":..}
    predictions = predictions_block["predictions"]   # list of detected objects
 
    match = next(
        (p for p in predictions if p["class"].strip() == class_name), None
    )
    if match is None:
        available = [p["class"] for p in predictions]
        raise ValueError(
            f"Class '{class_name}' not found in result. Available: {available}"
        )
 
    points = [(p["x"], p["y"]) for p in match["points"]]
    model_w, model_h = img_meta["width"], img_meta["height"]
    return points, model_w, model_h
 
 
def _draw_capped_line(draw, p1, p2, color, width, cap_len):
    """Draws a line from p1 to p2 with small perpendicular end-cap ticks
    (a caliper look), regardless of the line's angle."""
    draw.line([p1, p2], fill=color, width=width)
 
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length      # unit vector along the line
    px, py = -uy, ux                       # perpendicular unit vector
    half = cap_len / 2
    for (cx, cy) in (p1, p2):
        a = (cx - px * half, cy - py * half)
        b = (cx + px * half, cy + py * half)
        draw.line([a, b], fill=color, width=width)
 
 
def draw_ear_calipers(
    image_path,
    api_result,
    output_path="ear_calipers.png",
    class_name="ear",
    pad_x_frac=0.55,      # left/right padding around ear, as a fraction of ear width
    pad_y_frac=0.20,      # top/bottom padding, as a fraction of ear height
    width_line_height_frac=0.20,  # where (top->bottom) the horizontal caliper sits
    line_color=(255, 255, 255),
    outline_color=None,   # set to e.g. (230,30,30) if you also want the red outline
    line_width=2,
    cap_len=14,
):
    """
    Generates a cropped caliper-measurement image with THREE measurement
    lines, matching the reference style:
 
        1. Vertical caliper (outside, left of the ear)  -> overall ear height
        2. Horizontal caliper (near the top of the ear) -> upper ear width
        3. Diagonal caliper (drawn through the ear)      -> true point-to-point
           length between the ear's topmost and bottommost points (ears are
           rarely perfectly vertical, so this is the "true axis" length)
 
    Saves the annotated crop to output_path and returns the measurements
    (in the ORIGINAL image's pixel scale).
    """
    # 1. Pull polygon points + the resolution the model actually ran at
    points, model_w, model_h = _extract_ear_points(api_result, class_name)
 
    # 2. Load the real photo and rescale polygon coords to match its size
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    scale_x, scale_y = W / model_w, H / model_h
    pts = [(x * scale_x, y * scale_y) for x, y in points]
 
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ear_w_px, ear_h_px = max_x - min_x, max_y - min_y
 
    # 3. Crop tightly around the ear with padding
    pad_x = ear_w_px * pad_x_frac
    pad_y = ear_h_px * pad_y_frac
    crop_box = (
        max(0, min_x - pad_x),
        max(0, min_y - pad_y),
        min(W, max_x + pad_x * 0.6),
        min(H, max_y + pad_y),
    )
    crop_box = tuple(int(v) for v in crop_box)
    cropped = img.crop(crop_box)
 
    # 4. Shift everything into cropped-image coordinates
    ox, oy = crop_box[0], crop_box[1]
    pts_c = [(x - ox, y - oy) for x, y in pts]
    top, bottom = min_y - oy, max_y - oy
    left, right = min_x - ox, max_x - ox
 
    draw = ImageDraw.Draw(cropped)
 
    # (optional) trace the ear outline from the polygon
    if outline_color is not None:
        draw.line(pts_c + [pts_c[0]], fill=outline_color, width=2, joint="curve")
 
    # 5. Vertical caliper = overall ear height, placed just left of the ear
    vx = left - 18
    _draw_capped_line(draw, (vx, top), (vx, bottom), line_color, line_width, cap_len)
 
    # 6. Horizontal caliper = width near the TOP of the ear
    hy = top + (bottom - top) * width_line_height_frac
    band = [p for p in pts_c if abs(p[1] - hy) < (bottom - top) * 0.05]
    if band:
        hx_left = min(p[0] for p in band)
        hx_right = max(p[0] for p in band)
    else:
        hx_left, hx_right = left, right
    _draw_capped_line(draw, (hx_left, hy), (hx_right, hy), line_color, line_width, cap_len)
 
    # 7. Diagonal caliper = true length axis, topmost point -> bottommost point
    #    (drawn through the ear itself, like the reference image)
    top_pt = min(pts_c, key=lambda p: p[1])
    bottom_pt = max(pts_c, key=lambda p: p[1])
    _draw_capped_line(draw, top_pt, bottom_pt, line_color, line_width, cap_len * 0.85)
 
    cropped.save(output_path)
 
    diag_len_px = math.hypot(
        (bottom_pt[0] - top_pt[0]) / scale_x if scale_x else 0,
        (bottom_pt[1] - top_pt[1]) / scale_y if scale_y else 0,
    ) * ((scale_x + scale_y) / 2)  # ~ back to original-image pixel scale
 
    return {
        "output_path": output_path,
        "ear_height_px": ear_h_px,          # vertical bounding-box height
        "ear_width_top_px": hx_right - hx_left,  # width at the top band
        "ear_diagonal_length_px": diag_len_px,   # true top->bottom point distance
        "crop_box": crop_box,
    }
 
 
measurements = draw_ear_calipers('images.jpg', result_2, 'output_path-2.jpg')
print(measurements)