"""Embeddings + FAISS index, with content-signature caching so re-runs skip re-embedding."""
import hashlib
import json

import faiss
import numpy as np

from . import config
from .llm_client import client, call_with_retries


def compute_text_signature(texts: list[str], sample_n: int = 50) -> str:
    head = "\n".join(map(str, texts[:sample_n]))
    sample = f"{head}\nN={len(texts)}"
    return hashlib.sha256(sample.encode("utf-8", errors="ignore")).hexdigest()


def _sanity_check_vectors(v: np.ndarray) -> None:
    assert v.dtype == np.float32
    assert np.isfinite(v).all()
    assert v.ndim == 2 and v.shape[0] > 0 and v.shape[1] > 0
    assert (np.linalg.norm(v, axis=1) > 0).all()


def embed_texts(texts: list[str], batch_size: int = 128) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        def _call():
            return client.embeddings.create(model=config.EMBED_MODEL, input=batch)

        resp = call_with_retries(_call, "embeddings.create")
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)


def load_or_create_embeddings(texts: list[str]) -> np.ndarray:
    sig = compute_text_signature(texts)
    if config.EMB_CACHE.exists() and config.META_CACHE.exists():
        meta = json.load(open(config.META_CACHE))
        if meta.get("n") == len(texts) and meta.get("model") == config.EMBED_MODEL and meta.get("sig") == sig:
            v = np.load(config.EMB_CACHE).astype(np.float32)
            _sanity_check_vectors(v)
            return v

    v = embed_texts(texts)
    _sanity_check_vectors(v)
    np.save(config.EMB_CACHE, v)
    json.dump({"n": len(texts), "model": config.EMBED_MODEL, "sig": sig}, open(config.META_CACHE, "w"))
    return v


def build_or_load_index(kb) -> faiss.Index:
    texts = kb["kb_text"].tolist()
    sig = compute_text_signature(texts)

    if config.FAISS_INDEX_PATH.exists() and config.FAISS_META_PATH.exists():
        meta = json.load(open(config.FAISS_META_PATH))
        if meta.get("sig") == sig and meta.get("n") == len(texts):
            return faiss.read_index(str(config.FAISS_INDEX_PATH))

    kb_vectors = load_or_create_embeddings(texts)
    faiss.normalize_L2(kb_vectors)
    index = faiss.IndexFlatIP(kb_vectors.shape[1])
    index.add(kb_vectors)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    json.dump({"sig": sig, "n": len(texts), "model": config.EMBED_MODEL}, open(config.FAISS_META_PATH, "w"))
    return index
