from dataclasses import dataclass
from os import getenv
from typing import Optional

from agno.models.openai.like import OpenAILike


@dataclass
class VLLM(OpenAILike):
    """
    Class for interacting with the xAI API.

    Attributes:
        id (str): The ID of the language model.
        name (str): The name of the API.
        provider (str): The provider of the API.
        api_key (Optional[str]): The API key for the xAI API.
        base_url (Optional[str]): The base URL for the xAI API.
    """

    id: str = "neuralmagic/Qwen2-1.5B-Instruct-quantized.w8a16"
    name: str = "VLLM"
    provider: str = "VLLM"
    api_key: Optional[str] = getenv("VLLM_API_KEY")
    base_url: str = "http://localhost:8000/v1"
