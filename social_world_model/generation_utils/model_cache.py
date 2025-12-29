"""Model loading and caching for offline inference."""

import logging
from typing import Optional

# Import only when needed to avoid dependency errors
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelCache:
    """Singleton cache for vLLM models to avoid reloading."""

    _instance: Optional["ModelCache"] = None
    _engines: dict[str, "AsyncLLMEngine"] = {}

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_engine(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        **kwargs: object,
    ) -> "AsyncLLMEngine":
        """
        Get or create an AsyncLLMEngine for the specified model.

        Args:
            model_name: HuggingFace model identifier
            trust_remote_code: Allow custom model code
            gpu_memory_utilization: GPU memory fraction to use
            max_model_len: Maximum sequence length
            dtype: Model data type (auto, float16, bfloat16, etc.)
            **kwargs: Additional vLLM engine arguments

        Returns:
            Initialized AsyncLLMEngine
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed")

        cache_key = f"{model_name}_{gpu_memory_utilization}_{max_model_len}_{dtype}"

        if cache_key not in self._engines:
            logger.info(f"Loading vLLM model: {model_name}")

            engine_args = AsyncEngineArgs(
                model=model_name,
                trust_remote_code=trust_remote_code,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                **kwargs,
            )

            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._engines[cache_key] = engine
            logger.info(f"Model loaded successfully: {model_name}")

        return self._engines[cache_key]

    def clear(self) -> None:
        """Clear all cached models."""
        self._engines.clear()


# Global cache instance
_model_cache = ModelCache()


async def get_vllm_engine(
    model_name: str,
    **kwargs: object,
) -> "AsyncLLMEngine":
    """Helper function to get vLLM engine from cache."""
    return await _model_cache.get_engine(model_name, **kwargs)
