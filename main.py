# main.py
from command import DogController
from dotenv import load_dotenv
from voice import voice_to_text
import time
import os
import openai
import json

load_dotenv()
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")  # Get from https://openrouter.ai/keys
)

dog = DogController()

available_methods = {
    "handstand": dog.handstand,
    "sing": dog.sing,
    "flip": dog.flip
}

ROBOT_METHODS = [
    {
        "type": "function",
        "function": {
            "name": "handstand",
            "description": "Make the robot dog do a handstand or stop doing a handstand",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean",
                        "description": "True to start handstand, False to stop"
                    }
                },
                "required": ["on"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sing",
            "description": "Make the robot dog play an audio/music file. .MP3 should be concatenated to the last word of the song title and any spaces in a song title should be replaced with an underscore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mp3_filename": {
                        "type": "string",
                        "description": "The name of the audio file to play (e.g., 'FettyWap.mp3')"
                    }
                },
                "required": ["mp3_filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flip",
            "description": "Make the robot dog do a flip in direction: front, back, left, right",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "The direction to flip the dog"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Make the robot dog do a flip in direction: front, back, left, right",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]

def interpret_command(user_input: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are a robot dog controller. Interpret user commands and call the appropriate function to control the robot."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            tools=ROBOT_METHODS,
            tool_choice="auto"  
        )
        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
                
            return {
                "function": function_name,
                "arguments": function_args,
                "raw_response": message.content
            }
        else:
            # AI responded with text, no function call
            return {
                "function": None,
                "arguments": None,
                "raw_response": message.content
            }   
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
        return None

def execute_robot_command(controller: DogController, command_result: dict):
    if not command_result:
        print("No valid command received")
        return
    
    function_name = command_result["function"]
    args = command_result["arguments"]
    
    if function_name == "handstand":
        on = args["on"]
        controller.handstand(on)
        print(f"Handstand: {'ON' if on else 'OFF'}")
        
    elif function_name == "sing":
        filename = args["mp3_filename"]
        controller.sing(filename)
        print(f"Playing: {filename}")

    elif function_name =="flip":
        direction = args["direction"]
        controller.flip(direction)
        print(f'flip direction: {direction}')
    elif function_name == "wave":
        controller.wave()
        print('Dog is waving')
    else:
        print(f"Unknown function: {function_name}")

def main():
    print('here: ', os.environ)
    dog.start()
    # give WebRTC time to connect
    time.sleep(20)
    # set the dog's mode (ai / normal / idle)
    dog.set_mode("ai")
    print("Ready! Commands:")
    print("handstand on")
    print("handstand off")
    print("sing <filename.mp3>")
    print("quit")

    while True:
        voice_input = voice_to_text()

        if not voice_input:
            continue
        command = interpret_command(voice_input)
        execute_robot_command(dog, command)

if __name__ == "__main__":
    main()
