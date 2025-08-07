import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq, GroqError

load_dotenv()

client = Groq(
    api_key= os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)