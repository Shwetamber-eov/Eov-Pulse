"""
models.py

Defines all supported LLMs and creates LangChain model instances.
"""
import os
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ============================================================
# Model Configuration
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single LLM."""
    # Internal name
    name: str
    # google | groq
    provider: str
    # Actual model name used by the provider
    model_name: str
    # Context window (tokens)
    context_window: int
    # Requests Per Minute
    rpm: int
    # Requests Per Day
    rpd: int
    # Tokens Per Minute
    # None if provider doesn't enforce it
    tpm: Optional[int] = None
    # Lower number = higher priority
    priority: int = 1
    # Temperature
    temperature: float = 0.0

# ============================================================
# Supported Models
# ============================================================

MODEL_CONFIGS = [
    # # ---------------- Google ---------------- #
    ModelConfig(
        name="gemini35lite",
        provider="google",
        model_name="gemini-3.5-flash-lite",   # Replace if using a newer model ID
        context_window=250_000,
        rpm=15,
        rpd=500,
        priority=1,
        temperature=0
    ),

    ModelConfig(
        name="gemini3.1lite",
        provider="google",
        model_name="gemini-3.1-flash-lite",   # Replace if using a newer model ID
        context_window=250_000,
        rpm=15,
        rpd=500,
        priority=2,
        temperature=0
    ),
    ModelConfig(
        name="gemini36",
        provider="google",
        model_name="gemini-3.6-flash",   # Replace if using a newer model ID
        context_window=250_000,
        rpm=5,
        rpd=20,
        priority=2,
        temperature=0
    ),
    ModelConfig(
        name="gemini35",
        provider="google",
        model_name="gemini-3.5-flash",   # Replace if using a newer model ID
        context_window=250_000,
        rpm=5,
        rpd=20,
        priority=1,
        temperature=0
    ),

    ModelConfig(
        name="llama3.3",
        provider="groq",
        model_name="llama-3.3-70b-versatile",   # Replace if using a newer model ID
        context_window=12_000,
        rpm=30,
        rpd=1_000,
        priority=3,
        temperature=0
    ),

    # ---------------- Gemma ---------------- #
    ModelConfig(
        name="gemma31",
        provider="google",
        model_name="gemma-4-31b-it",            # Replace with your desired Groq model
        context_window=16_000,
        rpm=30,
        rpd=14_400,
        tpm=16_000,
        priority=4,
        temperature=0
    ),

    ModelConfig(
        name="gemma26",
        provider="google",
        model_name="gemma-4-26b-it",            # Replace with your desired Groq model
        context_window=16_000,
        rpm=30,
        rpd=14_400,
        tpm=16_000,
        priority=4,
        temperature=0
    ),
]

# Factory
def create_llm(config: ModelConfig):
    """
    Create a LangChain LLM instance from ModelConfig.
    """

    if config.provider == "google":
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model=config.model_name,
            temperature=config.temperature,
        )

    if config.provider == "groq":
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=config.model_name,
            temperature=config.temperature,
        )
    raise ValueError(f"Unsupported provider: {config.provider}")

# Helpers
def get_model_config(name: str) -> ModelConfig:
    """
    Returns a model configuration by its internal name.
    """
    for config in MODEL_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown model: {name}")