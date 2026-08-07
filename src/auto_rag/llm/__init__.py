"""LLM layer: provider abstraction plus concrete Ollama/OpenAI clients."""

from auto_rag.llm.base import LLM
from auto_rag.llm.factory import build_llm
from auto_rag.llm.models import ChatMessage, ChatRole, LLMResponse
from auto_rag.llm.ollama import OllamaLLM
from auto_rag.llm.openai_compat import OpenAICompatLLM

__all__ = [
    "LLM",
    "build_llm",
    "ChatMessage",
    "ChatRole",
    "LLMResponse",
    "OllamaLLM",
    "OpenAICompatLLM",
]
