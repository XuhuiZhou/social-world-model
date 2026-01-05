"""Main generation router supporting both API and offline vLLM inference."""

from typing import TypeVar

from sotopia.generation_utils import agenerate as sotopia_agenerate

OutputType = TypeVar("OutputType")


async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: object,
    temperature: float | None = 0.7,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> OutputType:
    """
    Unified generation function supporting both API and offline vLLM models.

    Args:
        model_name: Model identifier. Prefix with "vllm/" for offline models
        template: Prompt template with {variables}
        input_values: Dictionary mapping template variables to values
        output_parser: Parser for structured output
        temperature: Sampling temperature
        structured_output: Use structured JSON output
        bad_output_process_model: Fallback model for reformatting
        use_fixed_model_version: Use pinned model versions

    Returns:
        Parsed output matching OutputType

    Examples:
        # API model (routes to sotopia)
        await agenerate("gpt-4o", template, values, parser)

        # Offline vLLM model
        await agenerate("vllm/microsoft/Phi-4-mini-instruct", template, values, parser)
    """
    # Route based on model name prefix
    if model_name.startswith("vllm/") or model_name.startswith("vllm@"):
        # Check if vLLM is available
        try:
            from social_world_model.generation_utils.vllm_backend import (
                agenerate_vllm,
            )
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: uv sync --extra training"
            ) from e

        return await agenerate_vllm(
            model_name=model_name,
            template=template,
            input_values=input_values,
            output_parser=output_parser,
            temperature=temperature,
            structured_output=structured_output,
        )
    else:
        # Route to sotopia for API-based models
        return await sotopia_agenerate(
            model_name=model_name,
            template=template,
            input_values=input_values,
            output_parser=output_parser,
            temperature=temperature,
            structured_output=structured_output,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
