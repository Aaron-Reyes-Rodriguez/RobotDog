# main.py
import os
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
import tensorflow as tf
import json

# -----------------------------
# Paths (script-relative)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_classifier.keras")
MAPPING_PATH = os.path.join(BASE_DIR, "class_indices.json")
IMG_SIZE = (160, 160)

# -----------------------------
# Cooldowns & thresholds
# -----------------------------
HEART_COOLDOWN = 2.0
last_heart_time = 0.0

CIRCLE_PATH_LEN = 60
CIRCLE_MIN_RADIUS = 40
CIRCLE_MIN_MOVEMENT = 90
CIRCLE_FULL_SWEEP_RATIO = 0.95
CIRCLE_COOLDOWN = 1.5
last_circle_time = 0.0

STABILITY_MOVEMENT_THRESHOLD = 0.025  # a bit more forgiving

THRESH = 0.35  # probability threshold (less strict)

# -----------------------------
# Load model & mapping
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found at {MODEL_PATH} — run train_model.py first")

model = tf.keras.models.load_model(MODEL_PATH)
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_seg = mp.solutions.selfie_segmentation
segmenter = mp_seg.SelfieSegmentation(model_selection=1)

# state
path = deque(maxlen=CIRCLE_PATH_LEN)
recent_wrist_positions = deque(maxlen=15)

# -----------------------------
# helpers
# -----------------------------
def is_pose_stable(lst):
    if len(lst) < 6:
        return False
    pts = np.array([p for (_, p) in lst])
    disp = np.linalg.norm(pts[-1] - pts[0])
    return disp < STABILITY_MOVEMENT_THRESHOLD

def extract_hands_only(frame, multi_hand_landmarks):
    if not multi_hand_landmarks:
        return None
    h, w = frame.shape[:2]
    seg = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = (seg.segmentation_mask > 0.3).astype(np.uint8)

    xs, ys = [], []
    for hand in multi_hand_landmarks:
        for lm in hand.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
    if not xs:
        return None

    x0, y0 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
    x1, y1 = min(max(xs) + 20, w), min(max(ys) + 20, h)

    crop = frame[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop.size == 0 or crop_mask.size == 0:
        return None
    hands_only = crop * crop_mask[:, :, None]
    hands_only = cv2.resize(hands_only, IMG_SIZE)
    return hands_only

def detect_circle(path_pts):
    if len(path_pts) < 25:
        return False
    pts = np.array(path_pts, dtype=np.float32)
    movement = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    if movement < CIRCLE_MIN_MOVEMENT:
        return False
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    if radius < CIRCLE_MIN_RADIUS:
        return False
    d = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    if np.std(d) > radius * 0.6:
        return False
    angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    unwrapped = np.unwrap(angles)
    sweep = abs(unwrapped[-1] - unwrapped[0])
    if sweep < (2*math.pi*CIRCLE_FULL_SWEEP_RATIO):
        return False
    diffs = np.diff(unwrapped)
    direction_changes = np.sum(np.sign(diffs[:-1]) != np.sign(diffs[1:]))
    if direction_changes > 6:
        return False
    return True

# -----------------------------
# MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # wrist positions for stability
        if results.multi_hand_landmarks:
            wrist = results.multi_hand_landmarks[0].landmark[0]
            recent_wrist_positions.append((time.time(), np.array([wrist.x, wrist.y])))
        else:
            recent_wrist_positions.clear()

        # draw
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # HEART detection (hands-only)
        if results.multi_hand_landmarks:
            crop = extract_hands_only(frame, results.multi_hand_landmarks)
            if crop is not None:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                img = preprocess_input(crop.astype(np.float32))
                img = np.expand_dims(img, 0)
                prob = float(model.predict(img)[0][0])
                pred_label = inv_map[1] if prob > THRESH else inv_map[0]

                now = time.time()
                if pred_label == "heart" and is_pose_stable(recent_wrist_positions):
                    if now - last_heart_time > HEART_COOLDOWN:
                        last_heart_time = now
                        cv2.putText(frame, f"HEART ({prob:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print("heart detected")

        # CIRCLE detection (index finger path)
        if results.multi_hand_landmarks:
            idx = results.multi_hand_landmarks[0].landmark[8]
            px, py = int(idx.x * w), int(idx.y * h)
            path.append((px, py))
            for i in range(1, len(path)):
                cv2.line(frame, path[i-1], path[i], (0,255,0), 2)

            now = time.time()
            if detect_circle(list(path)):
                if now - last_circle_time > CIRCLE_COOLDOWN:
                    last_circle_time = now
                    cv2.putText(frame, "CIRCLE DETECTED!", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                    print("Circle gesture triggered")
                    path.clear()
        else:
            path.clear()

        cv2.imshow("Gesture Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

