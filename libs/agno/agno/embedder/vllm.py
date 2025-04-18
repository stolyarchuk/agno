from dataclasses import dataclass
from os import getenv
from typing import Optional

from agno.embedder.openai import OpenAIEmbedder


@dataclass
class VLLMEmbedder(OpenAIEmbedder):
    id: str = "intfloat/multilingual-e5-base"
    dimensions: int = 768
    api_key: Optional[str] = getenv("VLLM_API_KEY")
    base_url: str = "http://localhost:8000/v1"
