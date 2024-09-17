from io import BytesIO
import base64
from PIL import Image
import numpy as np
import os


def img2str(image):
    if isinstance(image, np.ndarray):
        buff = BytesIO()
        Image.fromarray(image).save(buff, format='PNG')
        result = base64.b64encode(buff.getvalue()).decode('utf-8')

    if isinstance(image, str) and os.path.isfile(image):
        with open(image, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
            result = encoded_string.decode('utf-8')

    result = 'data:image;base64,' + result
    return result

def apply_messages_template(image_b, response=None, demand="Translate inside table into latex code."):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_b}",
                },
                {"type": "text", "text": f'{demand}'},
            ],
        },

    ]
    if response is not None:
        response_value = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{response}"
                }
            ]
        }
        messages.append(response_value)
    return messages
