"""
ModelLoader — multi-provider LLM and embedding loader.

Supported providers:
  LLM       : groq (default), openai
  Embeddings: google (default), openai

Provider is selected via LLM_PROVIDER env var (default: groq).
Embedding provider is read from config.yaml embedding_model.provider.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config


class ApiKeyManager:
    _REQUIRED_KEYS = {
        "groq": ["GROQ_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
    }

    def __init__(self):
        self.keys: dict[str, str] = {}

        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")

        for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
            val = os.getenv(key)
            if val:
                self.keys[key] = val
                log.info(f"API key loaded: {key}")

    def get(self, key: str) -> str:
        val = self.keys.get(key) or os.getenv(key)
        if not val:
            raise DocumentPortalException(f"Missing API key: {key}", sys)
        return val

    def has(self, key: str) -> bool:
        return bool(self.keys.get(key) or os.getenv(key))


class ModelLoader:
    def __init__(self):
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("ModelLoader ready", config_keys=list(self.config.keys()))

    # ── Embeddings ─────────────────────────────────────────────────────────

    def load_embeddings(self):
        cfg = self.config.get("embedding_model")
        if not cfg:
            raise DocumentPortalException("Missing embedding_model in config.yaml", sys)

        provider = cfg.get("provider", "google")
        model_name = cfg.get("model_name")

        log.info("Loading embedding model", provider=provider, model=model_name)

        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
            )

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model_name or "text-embedding-3-small",
                openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
            )

        raise DocumentPortalException(
            f"Unsupported embedding provider: '{provider}'. Use 'google' or 'openai'.", sys
        )

    # ── LLM ────────────────────────────────────────────────────────────────

    def load_llm(self):
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        llm_cfg = self.config.get("llm", {})

        if provider not in llm_cfg:
            raise DocumentPortalException(
                f"LLM provider '{provider}' not found in config.yaml llm section.", sys
            )

        cfg = llm_cfg[provider]
        model_name = cfg["model_name"]
        temperature = cfg.get("temperature", 0)
        max_tokens = cfg.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise DocumentPortalException(
            f"Unsupported LLM provider: '{provider}'. Use 'groq' or 'openai'.", sys
        )
