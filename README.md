# Hand Mouse — Gesture-Controlled Mouse Using Computer Vision

Control your computer mouse entirely with your hand using real-time gesture recognition. No hardware required — just a webcam.

This is my first computer vision project, built to explore real-time hand tracking, landmark detection, and translating physical gestures into meaningful computer interactions.

---


## The Idea

I wanted to build something that sits at the intersection of computer vision and human-computer interaction. Most CV tutorials stop at detecting things — I wanted to go further and actually **do** something with the detection in real time.

The core challenge: how do you turn 21 hand landmarks into a full mouse controller with clicks, scrolling, and dragging — without the gestures conflicting with each other?

The answer ended up being a priority-based state machine where only one gesture can be active per frame, paired with relative tracking (like a trackpad) instead of absolute position mapping.

---

## Gesture Map

| Action | Gesture |
|---|---|
| **Move** | Open palm — palm center controls cursor |
| **Left Click** | Thumb + index finger pinch |
| **Double Click** | Thumb + index pinch twice quickly |
| **Right Click** | Thumb + middle finger pinch |
| **Middle Click** | Thumb + ring finger pinch |
| **Scroll** | Index + middle fingers held together, move up/down/left/right |
| **Drag** | Clench fist and move hand |

---

## How It Works

### Stack
- **MediaPipe** — detects 21 hand landmarks in real time using a pretrained ML model
- **OpenCV** — captures webcam frames and renders the landmark overlay
- **PyAutoGUI** — translates gesture decisions into actual mouse actions

### Architecture

Every frame goes through 4 steps:

1. **Extract landmarks** — get pixel positions of fingertips and knuckles from MediaPipe
2. **Calculate distances** — measure how close key finger pairs are to each other
3. **Decide gesture** — a priority chain picks one active gesture per frame (drag → left click → right click → scroll → move)
4. **Execute action** — only the winning gesture runs, preventing conflicts

### Key Design Decisions

**Relative tracking over absolute mapping**
The mouse moves based on how much your palm *moved* since the last frame, not where it is in the frame. This means you can lift your hand, reposition, and continue — exactly like a trackpad.

**State machine for gesture isolation**
Movement and clicking are mutually exclusive. When a click gesture is detected, mouse movement is frozen for that frame — preventing the natural hand wobble during a pinch from moving the cursor accidentally.

**Pinch release detection for double click**
Instead of using a timer to separate single from double clicks, the code tracks whether fingers actually *opened and closed again* between pinches — making double click feel natural and reliable.

**Fist detection via knuckle comparison**
A fist is detected when all four fingertip y-coordinates are greater than their corresponding middle knuckle y-coordinates — meaning all fingers are curled below their knuckles. This drives the drag gesture.

---

## Requirements

- macOS (currently Mac only — Windows/Linux support coming later)
- Python 3.x
- Webcam

---

## Installation & Running

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/hand_mouse.git
cd hand_mouse
```

**2. Install dependencies**
```bash
pip3 install opencv-python mediapipe pyautogui numpy --break-system-packages
```

**3. Download the MediaPipe hand landmark model**
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**4. Run**
```bash
python3 hand_mouse.py
```

Press **Q** to quit.

> **Note:** macOS will prompt for camera access and accessibility permissions on first run. Grant both — accessibility permissions are required for PyAutoGUI to control the mouse.

---

## Known Limitations

- **Mac only** — PyAutoGUI's mouse control behaves differently across platforms; cross-platform support is a future goal
- **Drag latency** — due to how macOS handles synthetic mouse events, drag updates are committed at release rather than in real time
- **Tracking dropouts** — MediaPipe may lose the hand briefly in low light or during fast movements; good lighting and a plain background improve reliability significantly
- **Gesture conflicts** — some hand shapes (e.g. right click with thumb + middle) can cause the tracker to misread adjacent fingers; being deliberate and keeping fingers clearly separated helps

---

## 📁 Project Structure

```
hand_mouse/
├── hand_mouse.py          # Main script
├── hand_landmarker.task   # MediaPipe model file (downloaded on first run)
└── README.md
```

---

## 🔮 What's Next

This project is intentionally kept as a script rather than a packaged app — the focus was learning the CV fundamentals. Future improvements could include:

- Windows/Linux support
- Configurable sensitivity and thresholds via a settings file
- Packaged as a menubar app for Mac

---

