import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Store the last N fingertip points
path = deque(maxlen=40)


# --- Circle Detection Function ---
def is_circle(points, tolerance=0.25):
    if len(points) < 10:
        return False
    
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]

    # Circle fit (least squares)
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    from scipy.optimize import leastsq
    center, _ = leastsq(f, (x_m, y_m))
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = np.mean(Ri)

    # Circularity error
    error = np.std(Ri) / R

    return error < tolerance


# --- Main Program ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Index fingertip landmark = 8
            lx = int(hand.landmark[8].x * w)
            ly = int(hand.landmark[8].y * h)
            path.append((lx, ly))

            # Draw trail
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 3)

            # CIRCLE DETECTED
            if is_circle(path):
                cv2.putText(frame, "CIRCLE DETECTED!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                print("circle")  # example action

                path.clear()  # reset so it doesn't keep firing repeatedly

        cv2.imshow("Circle Gesture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

