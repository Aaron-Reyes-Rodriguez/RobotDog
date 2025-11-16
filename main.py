# main.py
from command import DogController
from voice import voice_to_text

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
        voice_input = voice_to_text()

        # --------------------------
        # Handstand
        # --------------------------
        if voice_input.lower() == "handstand on":
            c.handstand(True)

        elif voice_input.lower() == "handstand off":
            c.handstand(False)

        # --------------------------
        # Sing
        # --------------------------
        elif voice_input.lower().startswith("sing "):
            # extract filename after "sing "
            filename = voice_input[5:].strip()
            if filename:
                print(f"Playing: {filename}")
                c.sing(filename)
            else:
                print("Error: no filename given. Example: sing FettyWap.mp3")

        # --------------------------
        # Quit
        # --------------------------
        elif voice_input.lower() == "quit":
            print("Closing connection…")
            c.close()
            break

        else:
            print("Unknown command. Try: handstand on | handstand off | sing file.mp3 | quit")


if __name__ == "__main__":
    main()
