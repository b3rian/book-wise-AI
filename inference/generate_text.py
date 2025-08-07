import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq, GroqError

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client() -> Groq:
    """Initialize and return the Groq client with error handling."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY is not set in the environment.")
        raise EnvironmentError("Missing GROQ_API_KEY in environment.")
    
    try:
        return Groq(api_key=api_key)
    except GroqError as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

def generate_completion(
    prompt: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    role: str = "user",
    system_prompt: Optional[str] = None
) -> str:
    """Generate a completion from Groq API using a given prompt and model."""
    client = get_groq_client()
     
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": role, "content": prompt})

    try:
        logger.info("Sending request to Groq API...")
        response = client.chat.completions.create(
            messages=messages,
            model=model
        )
        result = response.choices[0].message.content
        logger.info("Response received successfully.")
        return result
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Groq Text Generator")
    parser.add_argument("prompt", type=str, help="The prompt to send to the model")
    parser.add_argument("--model", type=str, default="meta-llama/llama-4-scout-17b-16e-instruct", help="Model ID to use")
    parser.add_argument("--system_prompt", type=str, help="Optional system-level instruction")

    args = parser.parse_args()

    try:
        response = generate_completion(
            prompt=args.prompt,
            model=args.model,
            system_prompt=args.system_prompt
        )
        print("\n--- Response ---\n")
        print(response)
    except Exception as e:
        logger.exception("Failed to generate text completion.")