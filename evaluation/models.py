"""OpenAI-compatible chat completion client and provider selection."""

import logging
from copy import copy
from typing import Optional, Dict, Any
import backoff
from dataclasses import dataclass
from evaluation.config import settings


@dataclass
class ModelResponse:
    """Simple response object with a text attribute."""

    text: str


class OpenAICompatibleEndpoint:
    """
    Simple wrapper for endpoints that use the OpenAI chat completions protocol
    (vLLM, Together AI, Gemini, Hugging Face Inference Providers).
    Exposes an `acomplete` method returning an object with a `.text` attribute.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float,
        request_timeout: float = 600,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the client with a custom API base URL."""
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.extra_body = extra_body or {}

        # Create an OpenAI client
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=self.api_base, api_key=self.api_key, timeout=self.request_timeout
        )

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Retry on any exception
        max_tries=10,  # Maximum number of retry attempts
        max_time=60,  # Maximum total time to retry for (in seconds)
        factor=2,  # Base for exponential backoff
    )
    async def acomplete(self, prompt: str) -> ModelResponse:
        """
        Complete a prompt asynchronously (sent as a single user message).
        Returns a ModelResponse object with a text attribute.
        Uses exponential backoff for automatic retries on failures.
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            # Set up parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False,
            }

            # Add extra_body if provided
            if self.extra_body:
                params["extra_body"] = self.extra_body

            # Make the API call with automatic retries via backoff decorator
            chat_completion = await self.client.chat.completions.create(**params)

            # Extract the generated text from the response
            generated_text = chat_completion.choices[0].message.content

            return ModelResponse(text=generated_text)
        except Exception as e:
            # Log the error and re-raise
            logging.error(f"Error in OpenAICompatibleEndpoint.acomplete: {str(e)}")
            raise


def apply_params_to_model(model, params: Optional[Dict[str, Any]] = None):
    """
    Apply parameter overrides to a model for a single completion.

    Returns the original model when there is nothing to override, otherwise a
    shallow copy with the given attributes replaced.
    """
    if not params:
        return model

    # Create a copy of the model to avoid modifying the original
    try:
        model_copy = copy(model)

        # Apply parameters directly to model attributes if they exist
        for param_name, param_value in params.items():
            if hasattr(model_copy, param_name):
                setattr(model_copy, param_name, param_value)

        return model_copy
    except Exception as e:
        # Log the error if copying fails
        logging.warning(f"Failed to apply parameters to model: {e}")
        raise e


def _require(value, env_name, provider):
    if not value:
        raise ValueError(
            f"{env_name} must be set in the environment (or in .env) for provider '{provider}'"
        )
    return value


def load_llm(model, provider, temperature, request_timeout=600):
    print(
        f"Using {provider} with model: {model} (temp={temperature}, timeout={request_timeout}s)"
    )
    if provider == "local":
        # Local OpenAI-compatible server (e.g. vLLM) at SCICHECK_API_BASE
        llm = OpenAICompatibleEndpoint(
            model=model,
            api_key=settings.model.local_api_key,
            api_base=settings.model.local_api_base,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    elif provider == "togetherai":
        # Models hosted on Together AI
        llm = OpenAICompatibleEndpoint(
            model=model,
            api_key=_require(
                settings.model.togetherai_api_key, "TOGETHERAI_API_KEY", provider
            ),
            api_base=settings.model.togetherai_api_base,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    elif provider == "google":
        # Google-specific configuration with thinking config
        google_extra_body = {
            "extra_body": {"google": {"thinking_config": {"include_thoughts": True}}}
        }

        llm = OpenAICompatibleEndpoint(
            model=model,
            api_key=_require(settings.model.google_api_key, "GOOGLE_API_KEY", provider),
            api_base=settings.model.google_api_base,
            temperature=temperature,
            request_timeout=request_timeout,
            extra_body=google_extra_body,
        )
    elif provider == "inference_providers":
        llm = OpenAICompatibleEndpoint(
            model=model,
            api_key=_require(settings.model.hf_api_key, "HF_API_KEY", provider),
            api_base=settings.model.inference_providers_api_base,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return llm
