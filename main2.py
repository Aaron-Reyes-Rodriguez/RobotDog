# main.py
from command import DogController
from voice import voice_to_text
import time
from follow_mode import follow_loop


def get_frame_from_go2(dog:DogController):
    # TODO: return latest BGR frame from your WebRTC video callback
    # For now, assume you have some global or queue updated by the connection.
    return dog.latest_frame

def main():
    c = DogController()
    c.start()

    # give WebRTC time to connect
    time.sleep(20)
    

    # set the dog's mode (ai / normal / idle)
    c.set_mode("ai")

    print("Ready! Commands:")
    print("  handstand on")
    print("  handstand off")
    print("  sing <filename.mp3>")
    print("  quit")

    while True:
        cmd = voice_to_text()

        # --------------------------
        # Handstand
        # --------------------------

        if cmd == "follow":
            follow_loop(c, get_frame_from_go2)
        
        elif cmd == "stop":
            c.move(0.0, 0.0)

        elif cmd == "backflip":
            c.flip("back")

        elif cmd.lower() == "handstand on":
            c.handstand(True)

        elif cmd.lower() == "handstand off":
            c.handstand(False)

        # --------------------------
        # Sing
        # --------------------------
        elif cmd.lower().startswith("sing "):
            # extract filename after "sing "
            filename = cmd[5:].strip()
            if filename:
                print(f"Playing: {filename}")
                c.sing(filename)
            else:
                print("Error: no filename given. Example: sing FettyWap.mp3")

        # --------------------------
        # Quit
        # --------------------------
        elif cmd.lower() == "quit":
            print("Closing connection…")
            c.close()
            break

        else:
            print("Unknown command. Try: handstand on | handstand off | sing file.mp3 | quit")


if __name__ == "__main__":
    main()
