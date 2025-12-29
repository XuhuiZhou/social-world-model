"""Generation utilities with offline vLLM support.

This module provides a drop-in replacement for sotopia.generation_utils
with additional support for offline vLLM inference.

Model name patterns:
- API models: Use sotopia backend (e.g., "gpt-4o", "together_ai/meta-llama/Llama-3-70b")
- vLLM models: Use offline backend (e.g., "vllm/microsoft/Phi-4-mini-instruct")
"""

from social_world_model.generation_utils.generate import agenerate
from sotopia.generation_utils import PydanticOutputParser, StrOutputParser

__all__ = [
    "agenerate",
    "PydanticOutputParser",
    "StrOutputParser",
]
