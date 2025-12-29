"""Simple test to verify vLLM integration."""

import asyncio
from social_world_model.generation_utils import agenerate, StrOutputParser


async def test_simple():
    """Test basic vLLM generation."""
    print("Testing vLLM offline inference...")
    print("Model: microsoft/Phi-4-mini-instruct")
    print("\nThis will take 30-60s on first run to load the model...")

    response = await agenerate(
        model_name="vllm/microsoft/Phi-4-mini-instruct",
        template="Answer this question in one sentence: {question}",
        input_values={"question": "What is 2+2?"},
        output_parser=StrOutputParser(),
        temperature=0.0,
    )

    print(f"\nResponse: {response}")
    print("\n✓ vLLM integration working!")


if __name__ == "__main__":
    asyncio.run(test_simple())
