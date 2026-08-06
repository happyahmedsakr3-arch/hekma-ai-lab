import os
from src.reader import analyze_image

API_KEY = os.getenv("OPENAI_API_KEY")

result = analyze_image(
    "samples/card.jpg",
    API_KEY
)

print(result)
