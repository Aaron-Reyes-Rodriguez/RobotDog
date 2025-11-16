# run_dog.py
import time
from command2 import DogController
from follow_mode import follow_loop, debug_video_loop 
from dotenv import load_dotenv
load_dotenv()


def get_frame_from_go2(dog: DogController):
    """
    Return the latest BGR frame that the WebRTC callback stored.
    DogController._async_setup() updates dog.latest_frame in the on_frame callback.
    """
    return dog.latest_frame


if __name__ == "__main__":
    dog = DogController()
    dog.start()

    print("[MAIN] Connecting to Go2 over WebRTC...")
    time.sleep(10)

    print("[MAIN] Setting base mode to 'normal'...")
    dog.set_mode("normal")

    # For now, just debug video:
    debug_video_loop(dog, lambda: get_frame_from_go2(dog))
    # Once video works, switch back to:
    # follow_loop(dog, lambda: get_frame_from_go2(dog), delay=0.5)
