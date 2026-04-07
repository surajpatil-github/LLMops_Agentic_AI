import os
import io
import types
import json
import shutil
import pathlib
import sys
import pytest

os.environ.setdefault("PYTHONPATH", str(pathlib.Path(__file__).resolve().parents[1] / "multi_doc_chat"))
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "google")

from fastapi.testclient import TestClient

# Ensure repository root is importable for `import main`
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main


@pytest.fixture
def client():
    return TestClient(main.app)


class _FakeSessionStore:
    """In-memory SessionStore drop-in for tests (no Redis needed)."""
    def __init__(self):
        self._data: dict = {}

    def create_session(self, sid):
        self._data[sid] = []

    def session_exists(self, sid):
        return sid in self._data

    def get_history(self, sid):
        return self._data.get(sid, [])

    def append_turn(self, sid, user_msg, assistant_msg):
        if sid not in self._data:
            self._data[sid] = []
        self._data[sid].append({"role": "user", "content": user_msg})
        self._data[sid].append({"role": "assistant", "content": assistant_msg})

    def delete_session(self, sid):
        self._data.pop(sid, None)

    def active_count(self):
        return len(self._data)

    def clear(self):
        self._data.clear()


@pytest.fixture
def clear_sessions(monkeypatch):
    """Replace the lazy session store in main with a fast in-memory fake."""
    fake = _FakeSessionStore()
    monkeypatch.setattr(main, "_session_store", fake)
    monkeypatch.setattr(main, "_get_session_store", lambda: fake)
    # Expose a SESSIONS-like dict so integration tests can seed data directly
    monkeypatch.setattr(main, "SESSIONS", fake._data, raising=False)
    fake.clear()
    main._rag_cache.clear()
    yield fake
    fake.clear()
    main._rag_cache.clear()


@pytest.fixture
def tmp_dirs(tmp_path: pathlib.Path):
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        # Point working directories used by app code to tmp ones by chdir
        os.chdir(tmp_path)
        yield {"data": data_dir, "faiss": faiss_dir}
    finally:
        os.chdir(cwd)


class _StubEmbeddings:
    def embed_query(self, text: str):
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.0, 0.1, 0.2] for _ in texts]

    def __call__(self, text: str):
        return [0.0, 0.1, 0.2]


class _StubLLM:
    def invoke(self, input):
        return "stubbed answer"


@pytest.fixture
def stub_model_loader(monkeypatch):
    # Patch both module paths to cover imports via `utils.model_loader` and `multi_doc_chat.utils.model_loader`
    #import utils.model_loader as ml_mod
    from multi_doc_chat.utils import model_loader as ml_mod

    from multi_doc_chat.utils import model_loader as ml_mod2

    class FakeApiKeyMgr:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x", "GOOGLE_API_KEY": "y"}

        def get(self, key: str) -> str:
            return self.api_keys[key]

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyMgr()
            self.config = {
                "embedding_model": {"model_name": "fake-embed"},
                "llm": {
                    "google": {
                        "provider": "google",
                        "model_name": "fake-llm",
                        "temperature": 0.0,
                        "max_output_tokens": 128,
                    }
                },
            }

        def load_embeddings(self):
            return _StubEmbeddings()

        def load_llm(self):
            return _StubLLM()

    monkeypatch.setattr(ml_mod, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(ml_mod2, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod2, "ModelLoader", FakeModelLoader)

    # Also patch the already-imported symbols used in modules under test
    import multi_doc_chat.src.document_ingestion.data_ingestion as di
    import multi_doc_chat.src.document_chat.retrieval as r
    monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(r, "ModelLoader", FakeModelLoader)
    yield FakeModelLoader


@pytest.fixture
def stub_ingestor(monkeypatch):
    import multi_doc_chat.src.document_ingestion.data_ingestion as di

    class FakeIngestor:
        def __init__(self, use_session_dirs=True, **kwargs):
            self.use_session = use_session_dirs
            self.session_id = "sess_test"

        def built_retriver(self, uploaded_files, **kwargs):
            return None

    monkeypatch.setattr(di, "ChatIngestor", FakeIngestor)
    monkeypatch.setattr(main, "ChatIngestor", FakeIngestor)
    yield FakeIngestor


@pytest.fixture
def stub_rag(monkeypatch):
    import multi_doc_chat.src.document_chat.retrieval as r

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
            self.retriever = retriever
            self._token_tracker = None

        def load_retriever_from_faiss(self, index_path, **kwargs):
            return None

        def invoke(self, user_input, chat_history=None):
            return "stubbed answer"

    monkeypatch.setattr(r, "ConversationalRAG", FakeRAG)
    monkeypatch.setattr(main, "ConversationalRAG", FakeRAG)
    yield FakeRAG