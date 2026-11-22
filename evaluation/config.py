"""Runtime configuration.

All credentials and endpoints are read from environment variables. A `.env`
file at the repository root is loaded if present (see `.env.example`);
variables already set in the environment take precedence.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
RESULTS_DIR = REPO_ROOT / "results"

load_dotenv(REPO_ROOT / ".env")


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value else default


class ModelSettings:
    """API keys and base URLs for the OpenAI-compatible providers."""

    def __init__(self):
        # Local OpenAI-compatible server that serves the SciCheck adapters
        # (e.g. vLLM, see scripts/serve_vllm.sh). No real key is needed.
        self.local_api_base = _env("SCICHECK_API_BASE", "http://127.0.0.1:8000/v1")
        self.local_api_key = _env("SCICHECK_API_KEY", "local")

        # Together AI (hosted models)
        self.togetherai_api_key = _env("TOGETHERAI_API_KEY")
        self.togetherai_api_base = _env(
            "TOGETHERAI_API_BASE", "https://api.together.xyz/v1"
        )

        # Google (Gemini) through its OpenAI-compatible endpoint
        self.google_api_key = _env("GOOGLE_API_KEY")
        self.google_api_base = _env(
            "GOOGLE_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # Hugging Face Inference Providers router
        self.hf_api_key = _env("HF_API_KEY")
        self.inference_providers_api_base = _env(
            "INFERENCE_PROVIDERS_API_BASE", "https://router.huggingface.co/v1"
        )


class Settings:
    def __init__(self):
        self.model = ModelSettings()


settings = Settings()
