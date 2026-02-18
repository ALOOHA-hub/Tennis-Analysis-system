# 🎾 AI Tennis Analysis System

An advanced Computer Vision pipeline for professional tennis match analysis. This system utilizes deep learning to track players and the ball, dynamically map the physical court, project 3D camera pixels onto a 2D radar, and calculate real-world physics such as player speed (km/h) and distance traveled.

## ✨ Key Features
* **Dual-Model Object Tracking:** Utilizes a standard YOLOv8x model for player detection and a custom fine-tuned YOLOv8x model specifically trained for high-speed tennis ball tracking.
* **TrackNet Court Detection:** Implements a custom 18-layer Convolutional Neural Network (CNN) to dynamically detect the 14 structural keypoints of a tennis court, completely independent of the video's resolution.
* **Spatial False-Positive Filtering:** Automatically filters out umpires, ball boys, and audience members by calculating the physical distance between detected people and the court boundaries.
* **2D Mini-Court Projection:** Maps 3D video coordinates to a mathematically perfect 2D top-down radar view using real-world tennis court dimensions.
* **Physics & Analytics Engine:** Calculates frame-by-frame player distance (meters) and windowed sprinting speed (km/h).
* **Developer "Stub" Caching:** Features a robust `.pkl` caching system that saves AI inference data. This allows for instantaneous pipeline re-runs during UI/UX testing without waiting for YOLO to process the video again.

## 📂 Project Structure
```text
Tennis Analysis system/
├── config.yaml              # Global configuration (paths, models, params)
├── main.py                  # Clean entry point
├── constants/               # System-wide magic numbers and colors
│   ├── detector_consts.py
│   ├── tracker_consts.py
│   └── visual_consts.py
├── core/                    # Core pipeline logic
│   ├── pipeline.py          # Main execution orchestrator
│   ├── analytics/
│   │   └── physics.py       # Speed and distance calculations
│   ├── annotation/          # OpenCV drawing utilities
│   │   ├── annotator.py
│   │   ├── entity_annotator.py
│   │   └── mini_court.py    # 2D radar map rendering
│   ├── detection/           # AI Model Wrappers
│   │   ├── court_detector.py# PyTorch TrackNet CNN
│   │   └── detector.py      # Ultralytics YOLO Wrapper
│   └── trackers/
│       └── tracker.py       # ByteTrack logic and spatial filtering
├── stubs/                   # Cached AI output for rapid development
├── utils/                   # Helpers (bbox math, config loader, logger)
├── models/                  # .pt and .pth model weights
└── data/                    # Input MP4s and Output AVIs
```

## 🚀 Installation & Setup

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```
*(Core dependencies: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `supervision`)*

**2. Download the Models:**
Place the following required model weights into the `models/` directory:
* `yolov8x.pt` (Standard YOLO)
* `yolov8x_ball_trained.pt` (Fine-tuned ball model)
* `model_tennis_court_det.pt` (TrackNet court keypoint model)

**3. Configure your paths:**
Open `config.yaml` and ensure your input video path and output destination are set correctly:
```yaml
paths:
  input_video: data/input/video_1.mp4
  output_video: data/output/output.avi
  unified_stub: stubs/tracks_stub.pkl
```

## 🎮 How to Run

Because the system is fully configuration-driven, running the pipeline is as simple as executing the main script:

```bash
python main.py
```

### Understanding the Execution Flow (The 5 Phases):
1. **AI Inference:** YOLOv8 extracts bounding boxes for all people and the tennis ball.
2. **Court Mapping:** The PyTorch CNN identifies the 14 intersections/corners of the tennis court.
3. **Filtering:** The Tracker interpolates missing ball frames and deletes bounding boxes for anyone standing outside the court (audience/umpires).
4. **Spatial Transformation:** The `MiniCourt` class uses mathematical scaling to project the players' feet and the ball's center onto a flat 2D tactical map.
5. **Analytics & Rendering:** The `PhysicsEngine` translates the 2D pixel movement into meters and km/h, and the `Annotator` draws the UI overlays onto the final video.

## 🧠 Architecture Highlights

* **No Hardcoded Resolutions:** The court detector dynamically calculates aspect ratios (`original_w / INPUT_WIDTH`), ensuring the 14 keypoints map perfectly whether the video is 720p, 1080p, or 4K.
* **Singleton Configuration:** The `ConfigLoader` utilizes the Singleton design pattern, ensuring that `config.yaml` is parsed exactly once and shared safely across all modules.
* **Modular Drawing:** The pipeline strictly separates data processing from video rendering. The `Annotator` is solely responsible for OpenCV `cv2` calls, keeping the `PhysicsEngine` and `Tracker` mathematically pure.