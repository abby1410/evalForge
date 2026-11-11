from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("EVALFORGE_PROVIDER", "demo").lower()
    model: str = os.getenv("EVALFORGE_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    judge_provider: str = os.getenv("JUDGE_PROVIDER", "same").lower()
    judge_model: str = os.getenv("JUDGE_MODEL", "")
    max_workers: int = max(1, int(os.getenv("EVALFORGE_MAX_WORKERS", "3")))
    timeout: int = max(10, int(os.getenv("EVALFORGE_REQUEST_TIMEOUT", "60")))
    pass_threshold: float = float(os.getenv("EVALFORGE_DEFAULT_PASS_THRESHOLD", "3.5"))
    regression_tolerance: float = float(os.getenv("EVALFORGE_REGRESSION_TOLERANCE", "0.25"))

settings = Settings()
