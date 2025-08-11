from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Required settings (will raise error if not provided)
    groq_api_key: str
    
    # Optional settings with defaults
    persist_dir: str = Field(default=r"D:\Documents\chromadb\nietzsche_db", description="Path to ChromaDB storage")
    collection_name: str = Field(default="nietzsche_books", description="ChromaDB collection name")
    model_name: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", description="Groq model to use")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create settings instance
settings = Settings()