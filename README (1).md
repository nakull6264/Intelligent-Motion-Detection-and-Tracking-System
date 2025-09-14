# 🎥🔍 Intelligent Motion Detection and Tracking System

A **real-time motion detection and object tracking system** using **YOLOv8**, **OpenCV**, **Flask**, and a **React frontend**. This project offers advanced video analysis with customizable parameters and a modern web interface. 🚀

## 🖼️ Screenshots

### Motion Detection with Pose Estimation
![image](https://github.com/user-attachments/assets/0237ecb2-e536-4d2b-8f3f-3c5d3d8051f3)


### React Frontend UI
![image](https://github.com/user-attachments/assets/2fbe4db6-3ccb-44c8-87ec-5ea759683d82)

### Multi-Object Detection (e.g., person and bicycle)
![image](https://github.com/user-attachments/assets/c92abbe8-c98b-4316-a376-6621e5d077a4)

## 📋 Overview

This application detects motion and tracks objects (e.g., person, car, cat) in a live video feed using the **YOLOv8 model**. It’s designed for real-time surveillance and monitoring, featuring:
- 🌐 Real-time video streaming via WebSocket
- ⚙️ Configurable detection thresholds (confidence, IoU)
- 🎨 Motion filtering with background subtraction
- 📱 Responsive web UI for visualization and control

## ✨ Features

- 🕵️‍♂️ Detects and tracks multiple object classes
- 🎚️ Adjustable duration, confidence, and IoU thresholds
- 📊 Logs detection data for post-run analysis
- 📹 Fallback to video file if webcam is not detected
- 🔍 Pose detection support with real-time keypoint mapping

## 🛠️ Prerequisites

- Python 3.12 🐍
- Node.js (for React frontend, optional for Flask+HTML) 🌐
- Git 📦

---

## 🏁 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Pranavv1039/Intelligent-Motion-Detection-and-Tracking-System.git
cd Intelligent-Motion-Detection-and-Tracking-System 
```

---

###  2. Set Up the Python Environment
```bash

python -m venv venv
venv\Scripts\activate        # On Windows
# Or: source venv/bin/activate  # On Linux/Mac
```

Install dependencies:

```bash
pip install flask flask-socketio ultralytics opencv-python numpy
```
Download the YOLOv8 model (e.g., yolov8m.pt) from Ultralytics and place it in the project root.
---

### 3. Set Up the Frontend
Basic Flask UI:
Use the default index.html in the templates/ folder.

React UI (optional):
```bash
npx create-react-app surveillance-app
cd surveillance-app
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```
Update tailwind.config.js and src/index.css accordingly, and replace src/App.js with the provided UI code.

---

### 4. Provide Video Source
🎥 Connect a webcam, or

📂 Place a video file (e.g., sample_video.mp4) in the root directory.

---

### 🚀 Usage
Backend (Flask)

```bash
python app.py
```
If using a custom video file:
```
python app.py --video D:\\pp1\\sample_video.mp4
```

---

### Frontend
Basic:
Open templates/index.html in a browser (Flask server must be running).
React:
```bash

cd surveillance-app
npm start
```
Visit http://localhost:3000 (React) or http://127.0.0.1:5000/ (Flask).

---

### 🎛️ Controls
Set Duration, Confidence Threshold, Motion Threshold

Switch between YOLOv8 models

View:

- Motion pixels

- Objects detected

- Bounding boxes

- Keypoint pose estimation

---

### ⚙️ Configuration
You can tweak parameters inside app.py:

```
CONFIG = {
    "MODEL_PATH": "yolov8m.pt",
    "CONFIDENCE_THRESHOLD": 0.4,
    "DURATION_SECONDS": 30,
    "IOU_THRESHOLD": 0.5,
    "CLASSES_OF_INTEREST": [0, 2],  # person, car
}
```
---

### 🛑 Troubleshooting:
- Issue	Fix
- Webcam not detected	Use a video file fallback
- 400 Error	Ensure index.html is in templates/
- WebSocket not connecting	Flask server must be running (localhost:5000)
- Permission denied	Run terminal or code as administrator
- Push failed to GitHub	Run git push -u --force origin main after setting remote

---

###  📢 Contributing
💡 Want to improve the project? Contributions are welcome! Fork the repo, create a branch, and submit a PR.

---

### 📜 License
This project is licensed under the MIT License – use freely with credit.

---

### ⏰ Version
Initial Release: June 24, 2025

Author: PRANAV

---

⭐ If you found this helpful, don't forget to star this repo! 🌟
Let me know if you want any modifications! 🚀🔥
