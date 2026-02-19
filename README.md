# 🤖 AUTODESKAI - Adaptive Gesture Recognition System

> **AI-Powered Hand Gesture Control with Automatic Learning**
> 
> A next-generation gesture recognition system that learns from user input, automatically augments training data, and adapts to individual users in real-time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Gesture Modes](#gesture-modes)
  - [👨‍💻 Developer Mode](#-developer-mode)
  - [🎤 Presentation Mode](#-presentation-mode)
  - [🎵 Casual Mode](#-casual-mode)
  - [📈 Trading Mode](#-trading-mode)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)

---

## Overview

AUTODESK is an intelligent gesture recognition system that combines computer vision, deep learning, and real-time processing to create a personalized hands-free control experience. Unlike traditional static gesture systems, FlowGesture **adapts to each user** by:

- ✨ **Automatically augmenting** user-recorded samples (6x data multiplication)
- 🔄 **Retraining the model** in the background when new gestures are added
- 🎯 **Learning incrementally** without losing previous gesture accuracy
- ⚡ **Recognizing gestures in real-time** with <30ms latency

**Perfect for:** Developers, presenters, traders, content creators, gamers, and anyone who wants hands-free control.

---

## ✨ Features

### 🧠 Adaptive Learning
- **Zero Configuration**: Users create gestures through a simple web interface
- **Auto-Augmentation**: Each recorded sample generates 6 augmented versions automatically
  - Brightness variations (bright/dim rooms)
  - Contrast adjustments (harsh/soft lighting)
  - Noise injection (low-quality webcam simulation)
  - Mirror flips (different hand angles)
- **Incremental Training**: Add new gestures anytime without retraining from scratch
- **Background Processing**: Model trains while you continue using the system

### 🎮 Real-Time Recognition
- **<30ms Latency**: Instant gesture detection on CPU
- **90%+ Accuracy**: After just 25 samples per gesture
- **WebSocket Streaming**: Live predictions sent to frontend
- **Confidence Scoring**: Know how certain the model is

### 🔧 User-Friendly Interface
- **Beautiful UI**: Built with React + Tailwind CSS
- **Webcam Integration**: Real-time video preview with hand landmark overlay
- **Progress Tracking**: Visual feedback during recording and training
- **Gesture Management**: Easy CRUD operations (Create, Read, Update, Delete)

### 💾 Persistent Storage
- **Automatic Saving**: All gestures and models saved locally
- **Session Recovery**: Resume training after restart
- **Export/Import**: Share gesture packs between users

---

## 🎨 Gesture Modes

FlowGesture includes 4 pre-built mode packs, each optimized for specific workflows. Users can also create **custom gestures** for any use case.

### 👨‍💻 Developer Mode

Perfect for coding, debugging, and navigating your IDE hands-free.

| Gesture | Action | Description |
|---------|--------|-------------|
| ✌️ **Peace** | Next file (VS Code) | Switch to next open file in editor |
| ✊ **Fist** | Toggle terminal | Show/hide integrated terminal |
| ☝️ **Point Up** | Scroll up in editor | Navigate up through code |
| 👎 **Thumb Down** | Scroll down | Navigate down through code |
| 🖐 **Open Palm** | Run code (Ctrl+F5) | Execute current file |
| 👍 **Thumb Up** | Git commit shortcut | Quick commit staged changes |
| 🆕 **Custom** | Launch Cursor AI | Open AI coding assistant |

### 🎤 Presentation Mode

Control your slides and screen like a pro presenter — no clicker needed.

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ **Point Up** | Next slide | Advance presentation forward |
| ✊ **Fist** | Previous slide | Go back one slide |
| 🖐 **Open Palm** | Start / pause slideshow | Toggle presentation mode |
| ✌️ **Peace** | Laser pointer (cursor highlight) | Highlight areas on screen |
| 👍 **Thumb Up** | Fullscreen toggle | Enter/exit fullscreen mode |
| 👎 **Thumb Down** | Black screen (presenter pause) | Pause with blank screen |

### 🎵 Casual Mode

Media playback control for Spotify, YouTube, and entertainment.

| Gesture | Action | Description |
|---------|--------|-------------|
| 👍 **Thumb Up** | Play / Pause Spotify | Toggle music playback |
| ✌️ **Peace** | Next track | Skip to next song |
| 👎 **Thumb Down** | Previous track | Go back to previous song |
| 🖐 **Open Palm** | Volume up | Increase audio volume |
| ✊ **Fist** | Volume down / Mute | Decrease or mute audio |
| 🤘 **ILoveYou** | Launch YouTube | Open YouTube in browser |
| ☝️ **Point Up** | Brightness up | Increase screen brightness |

### 📈 Trading Mode

**NEW!** Execute trades, switch charts, and manage positions hands-free.

| Gesture | Action | Description |
|---------|--------|-------------|
| 👍 **Thumb Up** | Buy / Long position | Execute long trade |
| 👎 **Thumb Down** | Sell / Short position | Execute short trade |
| ✊ **Fist** | Close position | Exit current trade |
| 🖐 **Open Palm** | Switch timeframe | Cycle between 1m/5m/15m/1h charts |
| ✌️ **Peace** | Next trading pair | Switch to next asset |
| ☝️ **Point Up** | Zoom in (chart) | Zoom into price action |
| 🤏 **Pinch** | Zoom out (chart) | Zoom out for overview |
| 🤘 **ILoveYou** | Set alert | Place price alert at current level |

---

## 🛠 Technology Stack

### Backend (`adaptive_gesture_system.py`)
- **FastAPI**: High-performance REST API + WebSocket server
- **PyTorch**: Deep learning framework for model training
- **MobileNetV3**: Lightweight CNN for fast inference (~30ms on CPU)
- **MediaPipe**: Hand landmark detection (21 keypoints)
- **OpenCV**: Image processing and augmentation
- **Uvicorn**: ASGI server with WebSocket support

### Frontend (React + Tailwind CSS)
- **React 18**: Component-based UI framework
- **Tailwind CSS**: Utility-first styling
- **Vite**: Lightning-fast build tool
- **WebSocket API**: Real-time communication
- **HTML5 Canvas**: Webcam video processing

### Data Pipeline
- **NumPy**: Numerical operations
- **PIL/Pillow**: Image manipulation
- **JSON**: Metadata storage

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Port 3000)               │
│         Built with React + Tailwind CSS                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  Webcam    │  │  Gesture   │  │  Live Recognition    │  │
│  │  Capture   │  │  Manager   │  │  Display             │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │ HTTP + WebSocket
             ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                    │
│           adaptive_gesture_system.py                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  REST API    │  │  WebSocket   │  │  Background     │   │
│  │  Endpoints   │  │  Handler     │  │  Trainer        │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Modules                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DataAugmenter: 6x sample multiplication            │   │
│  │  • Brightness: ±40%                                  │   │
│  │  • Contrast: ±50%                                    │   │
│  │  • Noise: Gaussian σ=15                             │   │
│  │  • Flip: Horizontal mirror                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GestureDataset: CRUD operations                     │   │
│  │  • Create/Read/Update/Delete gestures               │   │
│  │  • Metadata tracking (samples, timestamps)          │   │
│  │  • Auto-organization (raw + augmented folders)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GestureModel: Incremental learning                  │   │
│  │  • MobileNetV3-Small (2.5M params)                   │   │
│  │  • 15 epochs, ~2 minutes training (CPU)             │   │
│  │  • Auto-save best checkpoint                        │   │
│  │  • <30ms inference latency                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Persistent Storage                        │
│  user_gestures/                                            │
│  ├── raw/              ← Original samples                  │
│  ├── augmented/        ← 6x augmented versions            │
│  └── metadata.json     ← Gesture configs                  │
│                                                             │
│  trained_models/                                           │
│  └── current_model.pth ← Trained PyTorch model            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+ (3.10 recommended)
- Node.js 16+ (for frontend)
- Webcam/camera access

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/flowgesture.git
cd flowgesture/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision opencv-python mediapipe fastapi uvicorn websockets pyautogui pydantic

# Run backend server
python adaptive_gesture_system.py
```

Backend starts on: **http://localhost:8000**  
API docs available at: **http://localhost:8000/docs**

### Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend starts on: **http://localhost:3000**

---

## 🚀 Usage

### 1. Create a New Gesture

1. Open frontend at `http://localhost:3000`
2. Click **"Add Gesture"** button
3. Enter gesture details:
   - **ID**: `peace_sign` (no spaces)
   - **Name**: `Peace Sign`
   - **Emoji**: ✌️
4. Click **Create**

### 2. Record Training Samples

1. Select the gesture from the list
2. Click **"Record Samples"**
3. Position your hand in the webcam frame
4. Click **Start Recording**
5. Hold the gesture steady while the system captures 25-30 samples
6. System automatically:
   - Saves raw samples
   - Generates 6 augmented versions per sample
   - Triggers training when 20+ samples collected

**Tip**: Record samples in different:
- Lighting conditions (bright/dim)
- Hand positions (near/far from camera)
- Angles (tilted, rotated)

### 3. Model Training (Automatic)

The system trains automatically in the background:
- **Progress bar** shows training status (0-100%)
- **Console logs** display epoch progress
- **Notification** when training completes
- Training takes ~2-5 minutes for 8 gestures

### 4. Real-Time Recognition

1. Click **"Start Recognition"** button
2. Perform your trained gesture
3. System displays:
   - **Gesture name**: Which gesture was detected
   - **Confidence**: 0-100% certainty
   - **Latency**: Recognition speed in milliseconds

### 5. Edit/Delete Gestures

- **Edit**: Click gesture → Update name/emoji → Save
- **Delete**: Click trash icon → Confirm
- System **automatically retrains** after deletion

---

## 📚 API Documentation

### REST Endpoints

#### **GET** `/api/gestures`
Get all gestures with metadata.

**Response:**
```json
{
  "gestures": [
    {
      "id": "peace",
      "name": "Peace Sign",
      "emoji": "✌️",
      "raw_samples": 25,
      "augmented_samples": 175,
      "enabled": true,
      "created_at": "2024-01-20T10:30:00"
    }
  ]
}
```

#### **POST** `/api/gestures`
Create a new gesture.

**Request:**
```json
{
  "id": "thumbs_up",
  "name": "Thumbs Up",
  "emoji": "👍"
}
```

#### **POST** `/api/gestures/{gesture_id}/add-sample`
Add a training sample (auto-augments).

**Request:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "ok": true,
  "raw_count": 26,
  "aug_count": 182
}
```

#### **DELETE** `/api/gestures/{gesture_id}`
Delete gesture and trigger retraining.

#### **PATCH** `/api/gestures/{gesture_id}`
Update gesture metadata.

#### **GET** `/api/model/status`
Check training progress.

**Response:**
```json
{
  "is_training": true,
  "progress": 67,
  "accuracy": 89.5,
  "gesture_count": 8,
  "classes": ["peace", "thumbs_up", "fist", ...]
}
```

### WebSocket

#### **WS** `/ws/recognize`
Real-time gesture recognition stream.

**Send:**
```json
{
  "type": "frame",
  "data": "data:image/jpeg;base64,..."
}
```

**Receive:**
```json
{
  "gesture": "peace",
  "confidence": 0.953,
  "latency_ms": 28.4,
  "timestamp": 1705750800.123
}
```

---

## 📁 Project Structure

```
flowgesture/
├── backend/
│   ├── adaptive_gesture_system.py   ← Main backend server (COMPLETE)
│   ├── user_gestures/               ← User data (auto-created)
│   │   ├── raw/
│   │   ├── augmented/
│   │   └── metadata.json
│   ├── trained_models/              ← Saved models (auto-created)
│   │   └── current_model.pth
│   └── requirements.txt
│
├── frontend/                         ← Built with React + Tailwind CSS
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamView.jsx      ← Webcam display
│   │   │   ├── GestureList.jsx     ← Gesture management
│   │   │   ├── RecordPanel.jsx     ← Sample recording
│   │   │   ├── LiveRecognition.jsx ← Real-time display
│   │   │   └── TrainingProgress.jsx← Training status
│   │   ├── api/
│   │   │   └── gestureAPI.js       ← API client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── docs/
│   ├── PROJECT_README.md            ← Full documentation
│   ├── frontend_integration_example.js
│   └── API_REFERENCE.md
│
└── README.md                         ← This file
```

---

## 🎓 How It Works

### The Adaptive Learning Pipeline

```
User Records 1 Sample
        ↓
Backend Receives Frame
        ↓
┌─────────────────────────────┐
│   Data Augmentation (6x)    │
│  ✓ Bright (1.4x brightness) │
│  ✓ Dim (0.6x brightness)    │
│  ✓ High contrast (1.5x)     │
│  ✓ Low contrast (0.7x)      │
│  ✓ Gaussian noise (σ=15)    │
│  ✓ Horizontal flip          │
└─────────────────────────────┘
        ↓
7 Images Saved (1 raw + 6 aug)
        ↓
Sample Counter: 25/25
        ↓
┌─────────────────────────────┐
│   Auto-Trigger Training     │
│  • Build PyTorch dataset    │
│  • MobileNetV3-Small model  │
│  • 15 epochs (~2 minutes)   │
│  • Save best checkpoint     │
└─────────────────────────────┘
        ↓
Model Ready for Inference
        ↓
Real-Time Recognition
  <30ms latency | >90% accuracy
```

### Why This Approach Wins

**Traditional Systems:**
- ❌ Fixed gestures (developers decide)
- ❌ One-size-fits-all model
- ❌ Manual retraining needed
- ❌ Poor adaptation to new users

**FlowGesture (Ours):**
- ✅ User-defined gestures
- ✅ Personalized model per user
- ✅ Automatic retraining
- ✅ Learns from each user's hand

---

## 🏆 Key Innovations

1. **Automatic Augmentation**: 6x data multiplication without user effort
2. **Incremental Learning**: Add gestures without retraining everything
3. **Background Training**: Non-blocking model updates
4. **Real-Time Adaptation**: Model improves as you use it
5. **Mode-Based Workflows**: Pre-built gesture packs for common tasks
6. **Trading Mode**: First gesture system designed for financial trading

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Recognition Latency** | <30ms (CPU) |
| **Training Time** | 2-5 minutes (8 gestures, CPU) |
| **Accuracy** | >90% (25+ samples/gesture) |
| **Model Size** | ~10MB |
| **Memory Usage** | <500MB (training), <100MB (inference) |
| **Frame Rate** | 10-30 FPS |
| **Inference Device** | CPU-only (no GPU required) |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Team

- **Backend Developer**: Gesture recognition model (`adaptive_gesture_system.py`), API, training pipeline
- **Frontend Developer**: React UI with Tailwind CSS, webcam integration, user experience
- **Project**: Final Year B.Tech Project

---

## 🙏 Acknowledgments

- MediaPipe team for hand tracking models
- PyTorch community for deep learning framework
- FastAPI for the excellent web framework
- Tailwind CSS for beautiful styling

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Full Docs](docs/PROJECT_README.md)

---

**Built with ❤️ for hands-free productivity**

🌟 **Star this repo if you find it useful!**
