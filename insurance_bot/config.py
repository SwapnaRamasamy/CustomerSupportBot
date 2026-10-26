from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    pinecone_api_key: str = ""
    pinecone_index: str = "car-insurance-nomic"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    chat_model: str = "qwen3:4b-instruct"
    chat_timeout: float = Field(default=300, gt=0)
    chat_context_size: int = Field(default=8192, ge=2048)
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    embedding_dimensions: int = Field(default=768, gt=0)
    customer_support_phone: str = ""
    retrieval_k: int = Field(default=6, ge=1, le=20)
    semantic_min_score: float = Field(default=0.35, ge=-1, le=1)
    data_dir: Path = ROOT / "data"

    def require_credentials(self):
        if not self.pinecone_api_key:
            raise ValueError("Set PINECONE_API_KEY in .env.")
