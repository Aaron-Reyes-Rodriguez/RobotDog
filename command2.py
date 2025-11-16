
from __future__ import annotations
import asyncio
import threading
import time
import cv2
import numpy as np
from queue import Queue
import traceback


#from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
#from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

from go2_webrtc_connect.go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_connect.go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD



CMD_RATE_HZ = 1           # Move publish rate


class DogController: 
    def __init__(self): 
        self.loop = asyncio.new_event_loop()
        self.conn: Go2WebRTCConnection | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self.latest_frame = None
        

    # ------- PUBLIC 
    def start(self): 
        self._thread.start()

    def set_mode(self, name: str):
        self._schedule(self._publish(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": name}}
        ))

    def handstand(self, on: bool):
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandOut"], "parameter": {"data": on}}
        ))

    '''
    async def move(self): 
        dt = 1.0 / CMD_RATE_HZ

        for _ in range(2): 
            await asyncio.sleep(dt)
            await self._publish(
                RTC_TOPIC["RT_MOD"],
                {"api_id": SPORT_CMD["Move"],
                 "parameter": {"x": 1, "y": 0, "z": 0}}
            )
    '''

    def flip(self, direction: str):
        api = SPORT_CMD[{
            "front": "FrontFlip",
            "back":  "BackFlip",
            "left":  "LeftFlip",
            "right": "RightFlip"
        }[direction]]
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": api, "parameter": {"data": True}}
        ))
    
    def wave(self,direction: str):
        api = SPORT_CMD[{
            "circle": "Hello"
        }[direction]]
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": api, "parameter": {"data": True}}
        ))
    
    def move(self, forward: float, turn: float):
        params = {
            "vx": forward,
            "vy": 0.0,
            "vyaw": turn,
        }
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": params,
            }
        ))
 


    #-----------Audio test----

    def sing(self, mp3_filename: str = "Fetty Wap.mp3"):
        self._schedule(self._play_audio(mp3_filename))

    async def _play_audio(self, filename: str):
        from aiortc.contrib.media import MediaPlayer
        import os

        mp3_path = os.path.join(os.path.dirname(__file__), filename)
        player = MediaPlayer(mp3_path)
        audio_track = player.audio

        if self.conn:
            self.conn.pc.addTrack(audio_track)
            await asyncio.sleep(5)  # Adjust based on song duration or use player events


    def close(self):
        if self.conn:
            self._schedule(self.conn.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(2)

    # ------ PRIVATE 

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_setup())
        except Exception as e:
            print("[DOG] _async_setup FAILED:", repr(e))
            traceback.print_exc()
            return  # do not call run_forever if setup failed

        self.loop.run_forever()


    def _schedule(self, coro):
        """ Create a task for the thread to run"""
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))

    async def _publish(self, topic: str, payload: dict):
        """ Send the command to the dog!!!"""
        await self.conn.datachannel.pub_sub.publish_request_new(topic, payload)

    async def _async_setup(self):
        """Connect to the dog and set up video callbacks with debug logs."""
        print("[DOG] Creating WebRTC connection (LocalAP)...")
        self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

        print("[DOG] Connecting to Go2 over WebRTC...")
        await self.conn.connect()
        print("[DOG] WebRTC connected. PC state:", self.conn.pc.connectionState)

        # Show what transceivers we negotiated with the dog
        try:
            trans_info = [
                (t.kind, getattr(t, "direction", None))
                for t in self.conn.pc.getTransceivers()
            ]
            print("[DOG] Transceivers:", trans_info)
        except Exception as e:
            print("[DOG] Could not list transceivers:", e)

        @self.conn.pc.on("track")
        async def on_track(track):
            print(f"[DOG] Track received: kind={track.kind}")

            if track.kind == "video":
                print("[DOG] Video track registered, waiting for frames...")

                @track.on("frame")
                def on_frame(frame):
                    img = frame.to_ndarray(format="bgr24")
                    self.latest_frame = img
                    # Only print occasionally so it doesn’t spam:
                    # print("[DOG] Got a video frame.")







