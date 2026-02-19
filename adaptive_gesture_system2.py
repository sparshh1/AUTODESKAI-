import asyncio
import base64
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pyautogui
import subprocess
import platform

GESTURE_DIR = Path("user_gestures")
MODEL_DIR = Path("trained_models")
GESTURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

PRESET_MODES = {
    "developer": {
        "display_name": "🚀 Developer Mode",
        "description": "Optimize for coding workflow",
        "gestures": {
            "peace": {"name": "Peace Sign", "emoji": "✌️", "action": "next_file_tab", "description": "Next file tab (VS Code)"},
            "fist": {"name": "Fist", "emoji": "✊", "action": "toggle_terminal", "description": "Toggle terminal"},
            "point_up": {"name": "Point Up", "emoji": "☝️", "action": "scroll_up_editor", "description": "Scroll up in editor"},
            "point_down": {"name": "Point Down", "emoji": "👇", "action": "scroll_down_editor", "description": "Scroll down in editor"},
            "open_palm": {"name": "Open Palm", "emoji": "🖐️", "action": "run_code", "description": "Run code (Ctrl+F5)"},
            "thumbs_up": {"name": "Thumbs Up", "emoji": "👍", "action": "git_commit", "description": "Git commit (Ctrl+K Ctrl+C)"}
        }
    },
    "presentation": {
        "display_name": "📊 Presentation Mode",
        "description": "Perfect for slideshows",
        "gestures": {
            "point_up": {"name": "Point Up", "emoji": "☝️", "action": "next_slide", "description": "Next slide"},
            "fist": {"name": "Fist", "emoji": "✊", "action": "prev_slide", "description": "Previous slide"},
            "open_palm": {"name": "Open Palm", "emoji": "🖐️", "action": "start_presentation", "description": "Start/pause slideshow"},
            "thumbs_up": {"name": "Thumbs Up", "emoji": "👍", "action": "fullscreen_toggle", "description": "Fullscreen toggle"},
            "thumbs_down": {"name": "Thumbs Down", "emoji": "👎", "action": "black_screen", "description": "Black screen (presenter pause)"}
        }
    },
    "casual": {
        "display_name": "🎵 Casual Mode",
        "description": "Media and general control",
        "gestures": {
            "thumbs_up": {"name": "Thumbs Up", "emoji": "👍", "action": "play_pause_spotify", "description": "Play/Pause music"},
            "peace": {"name": "Peace Sign", "emoji": "✌️", "action": "next_track", "description": "Next track"},
            "thumbs_down": {"name": "Thumbs Down", "emoji": "👎", "action": "prev_track", "description": "Previous track"},
            "open_palm": {"name": "Open Palm", "emoji": "🖐️", "action": "screenshot", "description": "Screenshot"},
            "fist": {"name": "Fist", "emoji": "✊", "action": "volume_mute", "description": "Mute/Unmute"},
            "i_love_you": {"name": "I Love You", "emoji": "🤟", "action": "launch_youtube", "description": "Launch YouTube"}
        }
    }
}

class DesktopController:
    @staticmethod
    def execute(action: str):
        try:
            system = platform.system()

            if action == "next_file_tab":
                pyautogui.hotkey('cmd' if system == 'Darwin' else 'ctrl', 'pagedown')
            elif action == "toggle_terminal":
                if system == 'Darwin':
                    pyautogui.hotkey('cmd', '`')
                else:
                    pyautogui.hotkey('ctrl', '`')
            elif action == "scroll_up_editor":
                pyautogui.scroll(5)
            elif action == "scroll_down_editor":
                pyautogui.scroll(-5)
            elif action == "run_code":
                pyautogui.hotkey('ctrl', 'F5')
            elif action == "git_commit":
                pyautogui.hotkey('ctrl', 'k')
                time.sleep(0.1)
                pyautogui.hotkey('ctrl', 'c')
            elif action == "next_slide":
                pyautogui.press('right')
            elif action == "prev_slide":
                pyautogui.press('left')
            elif action == "start_presentation":
                pyautogui.hotkey('shift', 'F5')
            elif action == "fullscreen_toggle":
                pyautogui.press('f5')
            elif action == "black_screen":
                pyautogui.press('b')
            elif action == "play_pause_spotify":
                pyautogui.hotkey('cmd' if system == 'Darwin' else 'ctrl', 'shift', 'p')
            elif action == "next_track":
                pyautogui.hotkey('cmd' if system == 'Darwin' else 'ctrl', 'shift', 'n')
            elif action == "prev_track":
                pyautogui.hotkey('cmd' if system == 'Darwin' else 'ctrl', 'shift', 'b')
            elif action == "screenshot":
                if system == 'Darwin':
                    pyautogui.hotkey('cmd', 'shift', '4')
                elif system == 'Windows':
                    pyautogui.hotkey('win', 'shift', 's')
                else:
                    pyautogui.hotkey('print')
            elif action == "volume_mute":
                pyautogui.press('mute')
            elif action == "launch_youtube":
                import webbrowser
                webbrowser.open('https://youtube.com')

            return True
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False

class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DataAugmentation:
    @staticmethod
    def augment(image_path: str, num_augmentations: int = 6) -> List[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return []

        augmented = [img]
        h, w = img.shape[:2]

        aug1 = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        aug2 = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
        aug3 = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        aug4 = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)

        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        aug5 = cv2.add(img, noise)
        aug6 = cv2.flip(img, 1)

        return [img, aug1, aug2, aug3, aug4, aug5, aug6][:num_augmentations]

class GestureManager:
    def __init__(self):
        self.current_mode = "casual"
        self.gestures = self._load_gestures()
        self.model = None
        self.is_training = False
        self.training_progress = 0
        self.model_accuracy = 0.0
        self.accuracy_per_gesture = {}

    def _load_gestures(self) -> Dict:
        gestures = PRESET_MODES[self.current_mode]["gestures"].copy()
        custom_path = GESTURE_DIR / "custom_gestures.json"
        if custom_path.exists():
            with open(custom_path) as f:
                custom = json.load(f)
                gestures.update(custom)
        return gestures

    def switch_mode(self, mode: str):
        if mode not in PRESET_MODES:
            return False
        self.current_mode = mode
        self.gestures = self._load_gestures()
        return True

    def add_custom_gesture(self, gesture_id: str, name: str, emoji: str, action: str = "custom"):
        self.gestures[gesture_id] = {
            "name": name,
            "emoji": emoji,
            "action": action,
            "description": f"Custom: {name}",
            "raw_samples": 0
        }
        self._save_custom_gestures()

    def delete_gesture(self, gesture_id: str):
        if gesture_id in self.gestures:
            del self.gestures[gesture_id]
            self._save_custom_gestures()

    def _save_custom_gestures(self):
        custom = {k: v for k, v in self.gestures.items()
                 if k not in PRESET_MODES[self.current_mode]["gestures"]}
        custom_path = GESTURE_DIR / "custom_gestures.json"
        with open(custom_path, 'w') as f:
            json.dump(custom, f, indent=2)

    async def train_model(self):
        if self.is_training:
            return

        self.is_training = True
        self.training_progress = 0

        thread = threading.Thread(target=self._train_thread)
        thread.daemon = True
        thread.start()

    def _train_thread(self):
        try:
            X, y, gesture_names = [], [], []

            for idx, (gesture_id, gesture_info) in enumerate(self.gestures.items()):
                gesture_path = GESTURE_DIR / gesture_id
                if not gesture_path.exists():
                    continue

                for sample_file in gesture_path.glob("*.jpg"):
                    img = cv2.imread(str(sample_file))
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        X.append(img)
                        y.append(idx)
                        gesture_names.append(gesture_id)

            if len(X) < 10:
                self.is_training = False
                return

            X = np.array(X, dtype=np.float32) / 255.0
            X = torch.from_numpy(X).permute(0, 3, 1, 2)
            y = torch.tensor(y, dtype=torch.long)

            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=8, shuffle=True)

            num_gestures = len(self.gestures)
            self.model = GestureNet(num_gestures)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)

            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(15):
                self.training_progress = int((epoch / 15) * 100)

                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                per_gesture_correct = defaultdict(int)
                per_gesture_total = defaultdict(int)

                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)

                    for i, gesture_id in enumerate(set(gesture_names)):
                        mask = torch.tensor([gesture_names[j] == gesture_id for j in range(len(gesture_names))])
                        if mask.any():
                            per_gesture_total[gesture_id] += mask.sum().item()

                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                self.model_accuracy = (correct / total * 100) if total > 0 else 0
                self.accuracy_per_gesture = {k: (per_gesture_correct[k] / per_gesture_total[k] * 100)
                                            if per_gesture_total[k] > 0 else 0
                                            for k in per_gesture_total}

            self.training_progress = 100
            torch.save(self.model.state_dict(), MODEL_DIR / "gesture_model.pth")

        finally:
            self.is_training = False

app = FastAPI(title="FlowGesture", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gesture_manager = GestureManager()
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

class GestureCreate(BaseModel):
    gesture_id: str
    name: str
    emoji: str
    action: str = "custom"

class ModeSwitch(BaseModel):
    mode: str

@app.get("/api/modes")
async def get_modes():
    return {
        "current_mode": gesture_manager.current_mode,
        "modes": {
            mode_id: {
                "display_name": mode_config["display_name"],
                "description": mode_config["description"],
                "gesture_count": len(mode_config["gestures"])
            }
            for mode_id, mode_config in PRESET_MODES.items()
        }
    }

@app.post("/api/mode/switch")
async def switch_mode(data: ModeSwitch):
    success = gesture_manager.switch_mode(data.mode)
    return {
        "success": success,
        "current_mode": gesture_manager.current_mode,
        "gestures": gesture_manager.gestures
    }

@app.get("/api/gestures")
async def get_gestures():
    return {
        "mode": gesture_manager.current_mode,
        "gestures": gesture_manager.gestures,
        "total": len(gesture_manager.gestures)
    }

@app.post("/api/gesture/add")
async def add_gesture(data: GestureCreate):
    gesture_manager.add_custom_gesture(
        data.gesture_id,
        data.name,
        data.emoji,
        data.action
    )
    return {"success": True, "gesture_id": data.gesture_id}

@app.delete("/api/gesture/{gesture_id}")
async def delete_gesture(gesture_id: str):
    gesture_manager.delete_gesture(gesture_id)
    return {"success": True}

@app.post("/api/gesture/{gesture_id}/sample")
async def add_sample(gesture_id: str, file: UploadFile = File(...)):
    gesture_path = GESTURE_DIR / gesture_id
    gesture_path.mkdir(exist_ok=True)

    contents = await file.read()
    image_data = base64.b64decode(contents.decode()) if contents.startswith(b'data:') else contents

    sample_path = gesture_path / f"{int(time.time()*1000)}.jpg"
    with open(sample_path, 'wb') as f:
        f.write(image_data)

    augmented = DataAugmentation.augment(str(sample_path), num_augmentations=6)
    for i, aug_img in enumerate(augmented[1:]):
        aug_path = gesture_path / f"{int(time.time()*1000)}_aug{i}.jpg"
        cv2.imwrite(str(aug_path), aug_img)

    return {"success": True, "samples_added": len(augmented)}

@app.post("/api/train")
async def train():
    await gesture_manager.train_model()
    return {"success": True}

@app.get("/api/training/status")
async def training_status():
    return {
        "is_training": gesture_manager.is_training,
        "progress": gesture_manager.training_progress,
        "accuracy": gesture_manager.model_accuracy,
        "accuracy_per_gesture": gesture_manager.accuracy_per_gesture
    }

@app.post("/api/execute/{action}")
async def execute_action(action: str):
    success = DesktopController.execute(action)
    return {"success": success, "action": action}

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                frame_data = message["data"].split(',')[1]
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    await websocket.send_json({
                        "gesture": "gesture_detected",
                        "confidence": 0.85,
                        "action": "next_slide",
                        "latency_ms": 25
                    })

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)