# Face Analysis Project: Comprehensive Details

This document serves as a complete record of all the features, notebooks, algorithms, and web application components we have built so far in the Face Analysis project.

## 1. Core Technologies & Architecture
- **Backend Framework**: Python with Flask (`app.py`).
- **AI / Computer Vision**: Google MediaPipe Face Landmarker (478 dense 3D face mesh), OpenCV (`cv2`), and NumPy for geometric math.
- **Advanced Face Parsing**: HuggingFace Transformers (`jonathandinu/face-parsing` SegFormer) for semantic segmentation of facial features and background removal.
- **Frontend UI**: Pure HTML, CSS (modern glassmorphism, flexbox/grid layouts, CSS tooltips), and Vanilla JavaScript for dynamic interactions. No heavy frameworks, ensuring blazing fast performance.
- **LLM Integration**: Groq API integration (via `cheek_analysis.ipynb`) for generating dynamic, human-readable aesthetic descriptions based on geometric data.

---

## 2. Analytical Notebooks (R&D Phase)
Before integrating features into the web application, we mathematically modeled and tested them in Jupyter Notebooks to ensure accuracy.

### 👁️ Eye Analysis (`eye_analysis.ipynb`)
Extracts the eyes and calculates 7 key aesthetic metrics:
1. **Eye Tilt (Canthal Tilt)**: Calculates the angle between the inner and outer corners (Positive, Neutral, Negative).
2. **Eyelid Exposure**: Measures upper eyelid visibility (High, Moderate, Low).
3. **Sclera Color**: Samples pixel brightness/color to evaluate the whites of the eyes (White, Off-White, Discoloured).
4. **Under-Eye Health**: Compares luminance below the eye to the cheek to detect dark circles (Good, Moderate, Poor).
5. **Lower Eyelid Curvature (k)**: Uses a quadratic polynomial fit (`np.polyfit`) to mathematically measure the straightness or roundness of the lower lid.
6. **Eye Aspect Ratio (Almondness)**: Height-to-width ratio of the eye contour.
7. **Eye Spacing Ratio**: Intercanthal distance mapped against facial width.

### 🤨 Eyebrow Analysis (`eyebrow_analysis.ipynb`)
Evaluates the brow ridge and hair contour:
1. **Vertical Height**: Distance from the pupil to the eyebrow (mm).
2. **Elevation Ratio**: Arch height relative to the brow's baseline.
3. **Apex Angle**: The mathematical angle of the brow's arch.
4. **Position, Tilt, & Shape**: Classifications like "Low Set", "Arched", or "Straight".
5. **Virility / Dimorphism**: Evaluates masculine vs. feminine brow traits based on thickness and set height.

### 🦴 Additional Facial Features
- **Cheek Analysis (`cheek_analysis.ipynb`)**: Maps cheekbones and uses the Groq LLM API to generate dynamic textual descriptions of the cheek structure. (Syntax errors fully resolved).
- **Chin Analysis (`chin_analysis.ipynb`)**: Evaluates chin alignment (symmetry), chin width (breadth), and lower third height ratios.
- **Face Shape & Symmetry (`face_shape_analysis.ipynb`, `facial_symmetry.ipynb`)**: Foundational notebooks for mapping overall facial contours and bilateral symmetry.

---

## 3. Web Application Integration (Production Phase)
We have successfully ported the Eyebrow and Eye algorithms into the live Flask web server (`app.py`) and built a stunning, premium frontend UI based on specific design mockups.

### 🌐 Backend (`app.py`)
- **`/analyze_eyebrows` Endpoint**: 
  - Receives an image, runs MediaPipe, and calculates brow metrics. 
  - Includes robust error handling (e.g., `np.clip` to prevent `NaN` division by zero errors).
  - Crops the left and right eyebrows using rectangular bounding boxes, encodes them to base64, and sends them to the frontend.
- **`/analyze_eyes` Endpoint**:
  - Implements the 7 eye metrics finalized in the notebook.
  - Dynamically isolates and crops both eyes.
  - Uses highly tuned, sensitive classification thresholds to ensure users receive personalized results (avoiding everything defaulting to "Moderate").

### 🎨 Frontend UI (`templates/`)
- **Shared Design System**:
  - Clean, clinical, and premium aesthetics.
  - Sidebar navigation menu grouping facial features (Eyes, Eyebrows, Nose, Lips, Jaw, etc.).
  - Smooth micro-animations, custom slider tracks, and CSS hover tooltips.
- **`eyebrows.html`**:
  - Features a split-view layout.
  - Displays dynamic slider bars for Vertical Height, Elevation Ratio, and Apex Angle.
- **`eyes.html`**:
  - Perfectly replicates a specific UI mockup.
  - **Left Side**: A 2x2 grid displaying qualitative classes (Tilt, Exposure, Sclera, Health) with informative hover tooltips.
  - **Right Side**: An interactive, clickable slide carousel displaying quantitative metrics (Curvature, Almondness, Spacing).
  - **Dynamic JavaScript**: The titles and descriptions in the carousel dynamically rewrite themselves based on whether the user's mathematical results are below, within, or above standard anatomical ranges.

---

## 4. Code Health & Bug Fixes
- Ran a comprehensive sweep over all `.ipynb` and `.py` files to detect syntax errors.
- Resolved Windows encoding issues (`cp1252` vs `utf-8`) and Jupyter Magic command errors.
- Fixed complex math edge cases (e.g., replacing `scipy.interpolate.UnivariateSpline` with a stable `np.polyfit` to prevent curvature crashes on low-resolution eye contours).
- Reverted experimental masking (transparent polygons) back to clean, reliable rectangular JPEG bounding boxes based on user feedback.

---

## 5. Advanced Face Masking & Side-Profile Analysis
To generate highly accurate, anatomically flawless colored masks over specific facial features, we integrated advanced masking techniques:

- **Perfect Spline Interpolation**: Using `scipy.interpolate.splprep`, we generated perfectly smooth B-spline curves over the sharp, jagged MediaPipe landmarks. This enabled us to draw beautiful, sweeping gray masks over the **Cheeks** (`mediapipe_cheeks.ipynb`) and **Chin** (`mediapipe_chin.ipynb`).
- **Interactive File Uploads**: Integrated `tkinter.filedialog` to give users a native Windows pop-up for uploading their images securely and directly within the Jupyter Notebook environment.
- **Automated Side-Profile Jaw Masking (`mediapipe_jaw.ipynb`)**: 
  - Standard facial landmarkers (like MediaPipe) fail on pure 90-degree side profiles because they cannot detect both eyes.
  - **The Solution**: We built a fully automated geometric masking algorithm!
  - We use the `jonathandinu/face-parsing` AI model to extract the precise silhouette of the face (the "skin" class), which naturally ends exactly at the jawline and excludes the neck.
  - The algorithm automatically detects the earlobe and the front tip of the face (nose), then geometrically calculating the mouth area (approx. 65% of the distance).
  - It intersects this geometric bounding box with the AI skin mask to produce an automated, pixel-perfect curved jaw mask that flawlessly traces the jawline and avoids the lips, identically replicating the desired reference aesthetics!
