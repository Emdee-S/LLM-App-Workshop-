from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful and concise assistant. Answer briefly unless asked to elaborate.",
    )
    # Performance options
    num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    num_thread: int = int(os.getenv("OLLAMA_NUM_THREAD", str(os.cpu_count() or 4)))
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    # UI streaming batch settings (1 = update every token)
    stream_token_batch_size: int = int(os.getenv("STREAM_TOKEN_BATCH_SIZE", "1"))


def get_settings() -> Settings:
    return Settings()


