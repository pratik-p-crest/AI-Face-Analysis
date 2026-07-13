# AestheticAI: Facial & Ear Geometry Mapping Suite

AestheticAI is an advanced web application that utilizes computer vision (MediaPipe & Roboflow) and large language models (Groq LLM) to analyze facial and ear geometry, producing aesthetic structural reports.

## Features

- **Multi-Image Architecture**: A sleek, dark-mode web interface that accepts 3 photos for comprehensive analysis (Front Face, Left Profile, Right Profile).
- **Eyebrow Analysis**: Uses MediaPipe to map eyebrow landmarks. Calculates the inter-brow distance ratio and the brow arch ratio to determine shape dynamics (masculine/feminine aesthetics).
- **Cheek Structure**: Maps the Ogee curve and zygion markers using MediaPipe. Calculates lateral projection ratios and cheek height ratios to assess cheekbone prominence.
- **Ear Caliper Measurement**: Uses a Roboflow segmentation model to isolate the ear contour on profile images. Automatically draws vertical, horizontal, and diagonal measurement calipers to determine ear projection and dimensions.
- **AI LLM Reporting**: Compiles all the extracted mathematical ratios (width/height pixels, cheek projection ratios, etc.) and sends them to a Groq LLM API (`llama-3.1-8b-instant`), which generates a 3-part tailored aesthetic report.
- **Network Routing**: Configured to run locally on your network (`0.0.0.0:5000`), allowing you to upload photos and view results directly from your smartphone or tablet on the same Wi-Fi.

## Tech Stack

- **Backend**: Python, Flask, OpenCV (Image processing & geometric drawing), PIL
- **AI/ML**: MediaPipe (Face Landmarks), Roboflow HTTP API (Ear Segmentation), Groq API (LLM analysis)
- **Frontend**: HTML, CSS (Glassmorphism, animations, dark mode), Vanilla JavaScript

## Setup & Installation

1. **Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ROBOFLOW_API_KEY=your_roboflow_api_key_here
   ```

2. **Install Dependencies**:
   Ensure you have your virtual environment activated, then install the required Python packages (e.g. `flask`, `opencv-python`, `mediapipe`, `requests`, `groq`, `python-dotenv`, `pillow`).

3. **Run the Application**:
   Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be accessible locally at `http://127.0.0.1:5000` and across your Wi-Fi network at your local IP address (e.g., `http://192.168.x.x:5000`).

## How it Works

1. **Upload**: Drop your 3 photos into the glassmorphism UI.
2. **Process**: The backend uses MediaPipe for the front-facing photo and the Roboflow segmentation API for the profile photos.
3. **Annotate**: OpenCV and PIL automatically draw the precise contour lines, nodes, and measurement calipers onto the images.
4. **Report**: The raw metrics are sent to the Groq API to receive natural-language aesthetic insights.
5. **Display**: The UI updates dynamically to showcase the annotated Facial Map, Eyebrow close-ups, Ear Calipers, and the final 3-part report!
