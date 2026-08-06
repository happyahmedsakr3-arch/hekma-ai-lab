from openai import OpenAI
import os

MODEL = "gpt-5.6"

def get_client(api_key=None):
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )
