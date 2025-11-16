# collect_data.py
import os
import time
import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "dataset")
IMG_SIZE = (160, 160)

HEART_BURST = 100      # number of heart images when pressing H
NO_HEART_BURST = 25    # number of no_heart images when pressing N
DELAY_BETWEEN = 0.03   # seconds between frames saved

# -----------------------------
# Setup folders
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "heart"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "no_heart"), exist_ok=True)

# -----------------------------
# Mediapipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_seg = mp.solutions.selfie_segmentation
segmenter = mp_seg.SelfieSegmentation(model_selection=1)

hands_processor = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("DATA COLLECTOR")
print(" H : capture HEART burst ({} images)".format(HEART_BURST))
print(" N : capture NO_HEART burst ({} images)".format(NO_HEART_BURST))
print(" Q : quit")
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

def extract_hands_only(frame, multi_hand_landmarks):
    """Return hands-only image resized to IMG_SIZE (background black)."""
    if not multi_hand_landmarks:
        return None
    h, w = frame.shape[:2]

    # segmentation mask
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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_processor.process(rgb)

        preview = frame.copy()
        if results.multi_hand_landmarks:
            hands_img = extract_hands_only(frame, results.multi_hand_landmarks)
            if hands_img is not None:
                # small preview in top-left
                ph, pw = 200, 200
                preview[0:ph, 0:pw] = cv2.resize(hands_img, (pw, ph))

        cv2.putText(preview, "H=HEART (burst)   N=NO_HEART (burst)   Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Collect Data", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # HEART burst
        if key == ord('h') or key == ord('H'):
            print("Starting HEART burst (show heart pose)...")
            saved = 0
            for i in range(HEART_BURST):
                ret2, f2 = cap.read()
                if not ret2:
                    break
                f2 = cv2.flip(f2, 1)
                r2 = hands_processor.process(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                if not r2.multi_hand_landmarks:
                    # if no hands detected, skip saving this frame
                    time.sleep(DELAY_BETWEEN)
                    continue
                crop = extract_hands_only(f2, r2.multi_hand_landmarks)
                if crop is None:
                    time.sleep(DELAY_BETWEEN)
                    continue
                fname = os.path.join(SAVE_DIR, "heart", f"heart_{int(time.time()*1000)}_{i}.jpg")
                cv2.imwrite(fname, crop)
                saved += 1
                # show progress
                cv2.putText(crop, f"{saved}/{HEART_BURST}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Collect Data", cv2.resize(crop, (400,400)))
                cv2.waitKey(1)
                time.sleep(DELAY_BETWEEN)
            print(f"Heart burst done — saved {saved} images.")

        # NO_HEART burst
        if key == ord('n') or key == ord('N'):
            print("Starting NO_HEART burst (show non-heart poses)...")
            saved = 0
            for i in range(NO_HEART_BURST):
                ret2, f2 = cap.read()
                if not ret2:
                    break
                f2 = cv2.flip(f2, 1)
                r2 = hands_processor.process(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                # For no_heart we can save even if no hands detected, but prefer hands-only if present
                if r2.multi_hand_landmarks:
                    crop = extract_hands_only(f2, r2.multi_hand_landmarks)
                else:
                    # fallback: center crop
                    H, W = f2.shape[:2]
                    bw, bh = int(W*0.6), int(H*0.6)
                    x0, y0 = (W-bw)//2, (H-bh)//2
                    crop = cv2.resize(f2[y0:y0+bh, x0:x0+bw], IMG_SIZE)
                if crop is None:
                    continue
                fname = os.path.join(SAVE_DIR, "no_heart", f"no_heart_{int(time.time()*1000)}_{i}.jpg")
                cv2.imwrite(fname, crop)
                saved += 1
                cv2.putText(crop, f"{saved}/{NO_HEART_BURST}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Collect Data", cv2.resize(crop, (400,400)))
                cv2.waitKey(1)
                time.sleep(DELAY_BETWEEN)
            print(f"No-heart burst done — saved {saved} images.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands_processor.close()
    segmenter.close()



