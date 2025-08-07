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