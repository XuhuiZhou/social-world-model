"""vLLM backend implementation for offline inference."""

import json
import logging
import uuid
from typing import TypeVar

from pydantic import BaseModel

try:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from social_world_model.generation_utils.model_cache import get_vllm_engine

logger = logging.getLogger(__name__)
OutputType = TypeVar("OutputType")


async def agenerate_vllm(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: object,
    temperature: float | None = 0.7,
    structured_output: bool = False,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    **kwargs: object,
) -> OutputType:
    """
    Generate text using vLLM offline inference.

    Args:
        model_name: Model name in format "vllm/model/path" or "vllm@model/path"
        template: Prompt template
        input_values: Template variable values
        output_parser: Output parser
        temperature: Sampling temperature
        structured_output: Enable structured output (Pydantic models)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        **kwargs: Additional vLLM engine arguments

    Returns:
        Parsed output
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM not installed. Run: uv sync --extra training")

    # Extract actual model path from vllm/model/path or vllm@model/path
    if model_name.startswith("vllm/"):
        actual_model = model_name[5:]  # Remove "vllm/" prefix
    elif model_name.startswith("vllm@"):
        actual_model = model_name[5:]  # Remove "vllm@" prefix
    else:
        actual_model = model_name

    # Format prompt using simple string formatting
    # Add format instructions if using output parser
    format_instructions_dict = {}
    if hasattr(output_parser, "get_format_instructions"):
        format_instructions = output_parser.get_format_instructions()
        if "{format_instructions}" in template:
            # Include format instructions in input values
            format_instructions_dict = {"format_instructions": format_instructions}

    # Format the prompt with all values
    formatted_prompt = template.format(**input_values, **format_instructions_dict)

    # If format instructions weren't in template, append them
    if format_instructions_dict and "{format_instructions}" not in template:
        formatted_prompt += f"\n\n{format_instructions_dict['format_instructions']}"

    # Get or create vLLM engine
    engine = await get_vllm_engine(actual_model, **kwargs)

    # Handle structured output for Pydantic models
    structured_outputs_params = None
    if structured_output and hasattr(output_parser, "pydantic_object"):
        pydantic_obj = output_parser.pydantic_object
        if isinstance(pydantic_obj, type) and issubclass(pydantic_obj, BaseModel):
            # Use vLLM's structured outputs with JSON schema
            schema = pydantic_obj.model_json_schema()
            structured_outputs_params = StructuredOutputsParams(json=schema)
            logger.info(f"Enabling structured JSON generation with schema: {schema}")

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature if temperature is not None else 0.7,
        top_p=top_p,
        max_tokens=max_tokens,
        structured_outputs=structured_outputs_params,
    )

    # Generate text using vLLM
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id,
    )

    # Collect results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation produced no output")

    # Extract generated text
    generated_text = final_output.outputs[0].text

    # Parse output
    try:
        # If using structured output with guided JSON, parse as JSON directly
        if structured_output and hasattr(output_parser, "pydantic_object"):
            pydantic_obj = output_parser.pydantic_object
            if isinstance(pydantic_obj, type) and issubclass(pydantic_obj, BaseModel):
                # vLLM guided generation produces valid JSON
                json_data = json.loads(generated_text)
                parsed = pydantic_obj(**json_data)
                return parsed

        # Otherwise use the output parser's parse method
        parsed = output_parser.parse(generated_text)
        return parsed
    except Exception as e:
        logger.error(f"Failed to parse vLLM output: {e}")
        logger.error(f"Generated text: {generated_text}")
        raise
