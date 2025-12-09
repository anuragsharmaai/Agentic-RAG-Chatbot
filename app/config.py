import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_env: str = os.getenv("PINECONE_ENV", "us-east-1-aws")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "research-assistant")
    # Read as raw string from .env to avoid pydantic parsing errors, then parse below
    allowlisted_domains: str = os.getenv("ALLOWLISTED_DOMAINS", "")
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))

    class Config:
        env_file = ".env"

settings = Settings()

if "ALLOWLISTED_DOMAINS" in os.environ:
    domains_value = os.getenv("ALLOWLISTED_DOMAINS", "")
    domains_list = []
    for item in domains_value.split(","):
        stripped = item.strip()
        if len(stripped) > 0:
            domains_list.append(stripped)
    settings.allowlisted_domains = domains_list
