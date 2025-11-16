# follow_mode.py
import time
from tracker_ai import get_person_box_from_llm

def box_to_motion(box):
    cx = box["cx"]
    h  = box["h"]

    # Offset from center (-0.5 to 0.5)
    offset_x = cx - 0.5
    
    TURN_GAIN = 1.2
    FORWARD_GAIN = 0.8

    turn = -offset_x * TURN_GAIN

    desired_height = 0.32
    height_error = desired_height - h
    forward = height_error * FORWARD_GAIN

    # Limit speeds
    turn = max(min(turn, 1.0), -1.0)
    forward = max(min(forward, 0.6), -0.4)

    return forward, turn


def follow_loop(dog, get_frame, delay=0.4):
    """
    - dog: your DogController
    - get_frame: lambda returning latest OpenCV frame
    - delay: seconds between LLM vision calls (>0.35 recommended)
    """
    print("[FOLLOW] Starting follow mode...")

    try:
        while True:
            frame = get_frame()
            if frame is None:
                print("[FOLLOW] No frame yet.")
                time.sleep(delay)
                continue

            box = get_person_box_from_llm(frame)

            if box is None:
                print("[FOLLOW] No person detected. Stopping.")
                dog.move(0.0, 0.0)
            else:
                fwd, turn = box_to_motion(box)
                print(f"[FOLLOW] Move: forward={fwd:.2f}, turn={turn:.2f}")
                dog.move(fwd, turn)

            time.sleep(delay)

    except KeyboardInterrupt:
        print("[FOLLOW] Interrupted.")
        dog.move(0.0, 0.0)
