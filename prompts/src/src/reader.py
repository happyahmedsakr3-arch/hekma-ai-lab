from openai import OpenAI
from pathlib import Path
import base64

MODEL = "gpt-5.6"

def analyze_image(image_path, api_key):
    client = OpenAI(api_key=api_key)

    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()

    response = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "اقرأ هذه الصورة واستخرج جميع البيانات الموجودة بها."
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image}"
                }
            ]
        }]
    )

    return response.output_text
