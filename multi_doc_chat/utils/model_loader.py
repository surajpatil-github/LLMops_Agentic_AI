import os
import sys
from dotenv import load_dotenv

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class ApiKeyManager:
    def __init__(self):
        self.keys = {}

        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")

        for key in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
            val = os.getenv(key)
            if val:
                self.keys[key] = val
                log.info(f"Loaded {key}")

    def get(self, key: str) -> str:
        val = self.keys.get(key)
        if not val:
            raise DocumentPortalException(f"Missing API key: {key}", sys)
        return val


class ModelLoader:
    def __init__(self):
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    # -------------------------
    # EMBEDDINGS (GOOGLE ONLY)
    # -------------------------
    def load_embeddings(self):
        cfg = self.config.get("embedding_model")
        if not cfg:
            raise DocumentPortalException("Missing embedding_model in config.yaml", sys)

        provider = cfg.get("provider")
        model_name = cfg.get("model_name")

        log.info("Loading embedding model", provider=provider, model=model_name)

        if provider != "google":
            raise DocumentPortalException(
                "Only 'google' embeddings are supported",
                sys,
            )

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
        )

    # -------------------------
    # LLM (GROQ ONLY)
    # -------------------------
    def load_llm(self):
        provider = os.getenv("LLM_PROVIDER", "groq")
        llm_cfg = self.config.get("llm")

        if not llm_cfg or provider not in llm_cfg:
            raise DocumentPortalException(
                f"LLM provider '{provider}' not found in config.yaml",
                sys,
            )

        cfg = llm_cfg[provider]

        log.info("Loading LLM", provider=provider, model=cfg["model_name"])

        return ChatGroq(
            model=cfg["model_name"],
            api_key=self.api_key_mgr.get("GROQ_API_KEY"),
            temperature=cfg.get("temperature", 0),
        )
