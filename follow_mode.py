import time
import cv2
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

            # === Show camera feed ===
            cv2.imshow("Go2 Camera", frame)
            # Press 'q' in the window to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[FOLLOW] 'q' pressed. Stopping follow loop.")
                dog.move(0.0, 0.0)
                break

            # === LLM person detection + motion ===
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
    finally:
        cv2.destroyAllWindows()
        dog.move(0.0, 0.0)

def debug_video_loop(dog, get_frame, delay=0.05):
    """
    Simple loop to just display the video feed with no LLM, no movement.
    Press 'q' to quit.
    """
    print("[DEBUG] Starting video-only loop...")

    try:
        while True:
            frame = get_frame()
            if frame is None:
                print("[DEBUG] No frame yet.")
                time.sleep(delay)
                continue

            cv2.imshow("Go2 Camera (DEBUG)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DEBUG] 'q' pressed, exiting debug video loop.")
                break

            time.sleep(delay)

    except KeyboardInterrupt:
        print("[DEBUG] KeyboardInterrupt, exiting debug video loop.")
    finally:
        cv2.destroyAllWindows()
        dog.move(0.0, 0.0)