# AestheticAI: Facial & Ear Geometry Mapping Suite

AestheticAI is an advanced computer vision suite that utilizes Google MediaPipe, Roboflow, and Large Language Models (Groq) to analyze facial and ear geometry, producing aesthetic structural reports.

This repository is split into two main components:
1. **The Web Application** (`app.py`): A complete full-stack Flask app for multi-image analysis and AI reporting.
2. **The Research Notebooks**: A comprehensive collection of standalone Jupyter Notebooks designed for deep-dive mathematical facial analysis and geometric overlay generation.

---

## 📓 The Research Notebooks

The repository contains a suite of modular Jupyter Notebooks. Each notebook focuses on mathematically analyzing, mapping, and classifying specific aspects of human facial geometry from 2D images. 

All visualizations are generated using dynamic resolution scaling, synthetic padding, and OpenCV anti-aliasing for mathematically perfect geometric overlays.

### 1. 📐 Geometric & Structural Analysis
- **`face_shape_analysis.ipynb`**
  - **Function:** Classifies the overall geometric structure of the face (Oval, Square, Round, Heart, Oblong).
  - **Methodology:** Extracts 8 critical perimeter points and compares the ratios between Facial Length, Forehead Width, Midface Width, and Lower Third Width. Uses a synthetic forehead expansion algorithm to account for varying hairlines.
  - **Visualization:** Overlays a precise geometric octagon (facial bounding box), an ideal fitted ellipse, and dashed crosshair axes.
- **`facial_symmetry.ipynb`**
  - **Function:** Calculates a mathematical Facial Symmetry Score (out of 100).
  - **Methodology:** Establishes a "True Midline" vector from the Nasion to the Pogonion. Calculates the perpendicular distances and vertical alignment deviations for 9 paired facial regions (e.g., eyebrows, cheekbones, mouth corners).
  - **Visualization:** Draws the true midline with arrows and plots tiny, hollow white bullseye circles at key bilateral symmetry points.

### 2. 🧩 Regional Feature Analysis
- **`chin_analysis.ipynb`**
  - **Function:** Analyzes the geometry of the lower third of the face (Jawline and Chin).
  - **Methodology:** Maps the gonial angles and the Pogonion, evaluating the sharpness and width of the jaw.
  - **Visualization:** Automatically pads the image with a dark bottom panel to render a dynamic, color-coded legend detailing the annotations without overlapping features.
- **`cheek_analysis.ipynb`**
  - **Function:** Evaluates the prominence and structure of the cheekbones (Zygomatic arches).
  - **Methodology:** Measures the widest points of the midface relative to the jaw and forehead to determine cheekbone height and width.
- **`eyebrow_analysis.ipynb`**
  - **Function:** Analyzes eyebrow geometry and arch dynamics.
  - **Methodology:** Calculates the elevation angle, thickness, and arch apex of both the left and right eyebrows.

### 3. 🗺️ Comprehensive Facial Mapping
- **`2d_face_mapping.ipynb`**
  - **Function:** Generates a full 2D geometric map of the face.
  - **Methodology:** Extracts hundreds of key landmarks. Employs a synthetic coordinate expansion algorithm to extrapolate the top of the forehead (closing the facial dome above the eyebrows).
- **`3d_face_mapping.ipynb`**
  - **Function:** Generates an interactive, rotatable 3D topographic map of the face.
  - **Methodology:** Extracts the Z-depth coordinates from MediaPipe's 3D landmarker and plots them using `plotly.graph_objects`. 
  - **Output:** Saves a fully interactive `3d_face_map.html` file that can be opened in any web browser.

---

## 🌐 The Web Application

### Features
- **Multi-Image Architecture**: A sleek, dark-mode web interface that accepts 3 photos for comprehensive analysis (Front Face, Left Profile, Right Profile).
- **Eyebrow Analysis**: Uses MediaPipe to map eyebrow landmarks. Calculates the inter-brow distance ratio and the brow arch ratio to determine shape dynamics (masculine/feminine aesthetics).
- **Cheek Structure**: Maps the Ogee curve and zygion markers using MediaPipe. Calculates lateral projection ratios and cheek height ratios to assess cheekbone prominence.
- **Ear Caliper Measurement**: Uses a Roboflow segmentation model to isolate the ear contour on profile images. Automatically draws vertical, horizontal, and diagonal measurement calipers to determine ear projection and dimensions.
- **AI LLM Reporting**: Compiles all the extracted mathematical ratios (width/height pixels, cheek projection ratios, etc.) and sends them to a Groq LLM API (`llama-3.1-8b-instant`), which generates a 3-part tailored aesthetic report.
- **Network Routing**: Configured to run locally on your network (`0.0.0.0:5000`), allowing you to upload photos and view results directly from your smartphone or tablet on the same Wi-Fi.

### Tech Stack
- **Backend**: Python, Flask, OpenCV (Image processing & geometric drawing), PIL
- **AI/ML**: MediaPipe (Face Landmarks), Roboflow HTTP API (Ear Segmentation), Groq API (LLM analysis)
- **Frontend**: HTML, CSS (Glassmorphism, animations, dark mode), Vanilla JavaScript

## 🛠️ Setup & Installation

1. **Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ROBOFLOW_API_KEY=your_roboflow_api_key_here
   ```

2. **Install Dependencies**:
   Ensure you have your virtual environment activated, then install the required Python packages:
   ```bash
   pip install flask opencv-python mediapipe requests groq python-dotenv pillow matplotlib plotly jupyter nbformat
   ```

3. **Run the Application**:
   Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be accessible locally at `http://127.0.0.1:5000` and across your Wi-Fi network at your local IP address (e.g., `http://192.168.x.x:5000`).

4. **Run the Notebooks**:
   Start Jupyter Notebook to explore the standalone research modules:
   ```bash
   jupyter notebook
   ```
