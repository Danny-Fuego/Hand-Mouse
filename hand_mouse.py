import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import numpy as np
import pyautogui
import time
import math

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)
    print("Done!")

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

smooth_x, smooth_y = 0, 0
SMOOTHING = 5

prev_palm_x, prev_palm_y = None, None

pyautogui.FAILSAFE = False
is_dragging = False
pyautogui.PAUSE = 0

last_left_click = 0
CLICK_COOLDOWN = 0.2
click_pending = False
click_pending_time = 0
DOUBLE_CLICK_WINDOW = 0.4
pinch_released = True

last_right_click = 0
last_mid_click = 0

prev_scroll_y, prev_scroll_x = None, None
SCROLL_THRESHOLD = 48

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            h, w, _ = frame.shape

            CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            for connection in CONNECTIONS:
                start = connection.start
                end = connection.end
                x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            for landmark in hand:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            palm = hand[9]

            thumb_tip = hand[4]
            index_tip = hand[8]
            mid_tip = hand[12]
            ring_tip = hand[16]
            pinky_tip = hand[20]

            index_knuckle = hand[6]
            mid_knuckle = hand[10]
            ring_knuckle = hand[14]
            pinky_knuckle = hand[18]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            mid_x, mid_y = int(mid_tip.x * w), int(mid_tip.y * h)
            ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
            pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)

            index_kx, index_ky = int(index_knuckle.x * w), int(index_knuckle.y * h)
            mid_kx, mid_ky = int(mid_knuckle.x * w), int(mid_knuckle.y * h)
            ring_kx, ring_ky = int(ring_knuckle.x * w), int(ring_knuckle.y * h)
            pinky_kx, pinky_ky = int(pinky_knuckle.x * w), int(pinky_knuckle.y * h)


            # ── STEP 1: calculate all distances ─────────────────────
            left_dist = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            right_dist = math.sqrt((mid_x - thumb_x)**2 + (mid_y - thumb_y)**2)
            scroll_dist = math.sqrt((mid_x - index_x)**2 + (mid_y - index_y)**2)
            mid_dist = math.sqrt((thumb_x - ring_x)**2 + (thumb_y - ring_y)**2)

            is_fist = (
                index_y > index_ky and
                mid_y > mid_ky and
                ring_y > ring_ky and
                pinky_y > pinky_ky
            )

            # ── STEP 2: decide current gesture ──────────────────────
            if is_fist:
                current_gesture = "drag"
            elif left_dist < 48:
                current_gesture = "left_click"
            elif right_dist < 48:
                current_gesture = "right_click"
            elif scroll_dist < SCROLL_THRESHOLD:
                current_gesture = "scroll"
            elif mid_dist < 48:
                current_gesture = "mid_click"
            else:
                current_gesture = "move"

            # ── STEP 3: only move if gesture is "move" ──────────────
            if current_gesture == "move" or current_gesture == "drag":
                if prev_palm_x is not None:
                    delta_x = (palm.x - prev_palm_x) * screen_w * 2
                    delta_y = (palm.y - prev_palm_y) * screen_h * 2
                    if abs(delta_x) > 8 or abs(delta_y) > 8:
                        pyautogui.moveRel(delta_x, delta_y)
                prev_palm_x, prev_palm_y = palm.x, palm.y
            else:
                prev_palm_x, prev_palm_y = None, None

            # ── STEP 4: run the active gesture's action ─────────────
            if current_gesture == "drag":
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                cv2.putText(frame, "DRAG", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            elif current_gesture == "left_click" and pinch_released:
                pinch_released = False
                print(f"pinch detected | click_pending: {click_pending} | time since pending: {time.time() - click_pending_time:.2f}s")
                if click_pending and (time.time() - click_pending_time) < DOUBLE_CLICK_WINDOW:
                    print("DOUBLE CLICK FIRED")
                    pyautogui.doubleClick()
                    click_pending = False
                    cv2.putText(frame, "DOUBLE CLICK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    click_pending = True
                    click_pending_time = time.time()

                last_left_click = time.time()

            elif current_gesture == "right_click" and (time.time() - last_right_click) > CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_right_click = time.time()
                cv2.putText(frame, "RIGHT CLICK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif current_gesture == "mid_click" and (time.time() - last_mid_click) > CLICK_COOLDOWN:
                pyautogui.middleClick()
                last_mid_click = time.time()
                cv2.putText(frame, "MIDDLE CLICK", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif current_gesture == "scroll":
                if prev_scroll_y is not None:
                    delta_y = prev_scroll_y - mid_y
                    if abs(delta_y) > 8:
                        pyautogui.scroll(int(delta_y))
                        direction = "UP" if delta_y > 0 else "DOWN"
                        cv2.putText(frame, f"SCROLL {direction}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

                if prev_scroll_x is not None:
                    delta_x = prev_scroll_x - mid_x
                    if abs(delta_x) > 8:
                        pyautogui.hscroll(int(delta_x))
                        direction = "LEFT" if delta_x > 0 else "RIGHT"
                        cv2.putText(frame, f"SCROLL {direction}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

                prev_scroll_x, prev_scroll_y = mid_x, mid_y

            if current_gesture != "left_click":
                pinch_released = True
            
            if current_gesture != "drag" and is_dragging:
                pyautogui.mouseUp()
                is_dragging = False

            if current_gesture != "scroll":
                prev_scroll_x, prev_scroll_y = None, None

            # ── single-click timeout check — runs every frame regardless of gesture ──
            if click_pending and (time.time() - click_pending_time) > DOUBLE_CLICK_WINDOW:
                pyautogui.click()
                cv2.putText(frame, "LEFT CLICK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                click_pending = False

    else:
        prev_palm_x, prev_palm_y = None, None
        prev_scroll_x, prev_scroll_y = None, None

    frame = cv2.resize(frame, (480, 270))
    cv2.imshow("Hand Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()