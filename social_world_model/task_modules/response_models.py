"""Pydantic model for structured benchmark responses.

This unified model works for all ToM benchmarks (ToMi, FANToM, ConFAIDE, HiToM, COBRA-Frames, MMToM)
and enables vLLM's guided JSON generation for reliable parsing.
"""

from pydantic import BaseModel, Field, ConfigDict


class BenchmarkResponse(BaseModel):
    """Unified response format for all ToM benchmarks.

    This model ensures consistent output format and enables vLLM's
    guided JSON generation to avoid parsing failures.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description="Step-by-step reasoning explaining how you arrived at the answer. "
        "Explain your thought process clearly and logically."
    )
    answer: str = Field(
        description="The final answer to the question. "
        "Be concise and provide only the specific answer requested."
    )
