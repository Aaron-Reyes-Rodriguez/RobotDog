import os
import base64
import json
import requests
import cv2
from typing import Optional, Dict


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# pick a multimodal model name that supports images on OpenRouter
VISION_MODEL = "openai/gpt-4o-mini"  # example – check your OpenRouter models page


def _encode_image_b64(image_bgr) -> str:
    # Convert OpenCV BGR frame to JPEG base64
    _, buf = cv2.imencode(".jpg", image_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def get_person_box_from_llm(image_bgr) -> Optional[Dict[str, float]]:
    """
    Sends a frame to an OpenRouter vision model and asks for
    the main person's bounding box as normalized coordinates.
    Returns dict with keys cx, cy, w, h in [0,1] or None.
    """
    if OPENROUTER_API_KEY is None:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    image_data_url = _encode_image_b64(image_bgr)

    prompt = (
        "You are a vision model controlling a robot dog.\n"
        "Look at the image and find the main person that the robot should follow.\n"
        "If there is at least one person, return ONLY a JSON object with fields:\n"
        '{"cx": <center_x_norm>, "cy": <center_y_norm>, '
        '"w": <width_norm>, "h": <height_norm>}.\n'
        "All values must be floats between 0 and 1, normalized to the image size.\n"
        "If there is no person, return the JSON null value.\n"
        "Return JSON only, no explanation."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional attribution headers per docs:
        # "HTTP-Referer": "http://localhost", 
        # "X-Title": "Go2-Follow-Mode",
    }

    body = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            }
        ],
        # good practice for structured JSON:
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]

    # content should be JSON – but we still defend against weird output
    try:
        parsed = json.loads(content)
    except Exception:
        return None

    if parsed is None:
        return None

    # basic sanity checks
    try:
        cx = float(parsed["cx"])
        cy = float(parsed["cy"])
        w = float(parsed["w"])
        h = float(parsed["h"])
    except Exception:
        return None

    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return None

    return {"cx": cx, "cy": cy, "w": w, "h": h}