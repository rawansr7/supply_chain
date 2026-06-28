"""Turn description text into fixed-size embedding vectors, per ablation arm.

Backends (auto-selected, in order of preference):
  1. sentence-transformers  (local, free, good quality)  -> EMBEDDING_MODEL
  2. openai                  (text-embedding-3-large)
  3. hashing                 (deterministic bag-of-character-ngrams hashing trick;
                              dependency-free fallback so the pipeline always runs)

Embeddings are cached per (arm, backend) as parquet keyed by StockCode.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _embed_hashing(texts: list[str], dim: int) -> np.ndarray:
    """Signed-hashing of word + char-3gram tokens into `dim` buckets, L2-normed.

    Deterministic and dependency-free. Not semantically rich, but a fair, honest
    text-feature baseline (much better than nothing) and keeps CI runnable.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = (t or "").lower()
        tokens = t.split()
        tokens += [t[j:j + 3] for j in range(max(0, len(t) - 2))]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            bucket = h % dim
            sign = 1.0 if (h >> 8) % 2 else -1.0
            out[i, bucket] += sign
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def _embed_sentence_transformers(texts: list[str], dim: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.EMBEDDING_MODEL)
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    return _project(vecs, dim)


def _embed_openai(texts: list[str], dim: int) -> np.ndarray:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for k in range(0, len(texts), 256):  # batch
        chunk = [t or " " for t in texts[k:k + 256]]
        resp = client.embeddings.create(input=chunk, model=C.OPENAI_EMBEDDING_MODEL, dimensions=dim)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)


def _project(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Truncate or zero-pad to `dim` (3-large supports native dim; ST does not)."""
    if vecs.shape[1] == dim:
        return vecs
    if vecs.shape[1] > dim:
        return vecs[:, :dim]
    pad = np.zeros((vecs.shape[0], dim - vecs.shape[1]), dtype=np.float32)
    return np.hstack([vecs, pad])


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import sentence_transformers  # noqa: F401
        return "sentence-transformers"
    except Exception:
        return "hashing"


_BACKENDS = {
    "hashing": _embed_hashing,
    "sentence-transformers": _embed_sentence_transformers,
    "openai": _embed_openai,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def embed_descriptions(desc_df: pd.DataFrame, arm: str, backend: str | None = None,
                       force: bool = False) -> pd.DataFrame:
    """Return a DataFrame: StockCode + emb_0..emb_{dim-1} for one arm.

    `desc_df` must have columns ['StockCode', 'text'].
    """
    backend = _resolve_backend(backend or C.EMBEDDING_BACKEND)
    cache = C.EMBEDDINGS_DIR / f"embeddings_{arm}_{backend}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    texts = desc_df["text"].fillna("").astype(str).tolist()
    vecs = _BACKENDS[backend](texts, C.EMBEDDING_DIM)
    cols = [f"emb_{i}" for i in range(vecs.shape[1])]
    out = pd.DataFrame(vecs, columns=cols)
    out.insert(0, "StockCode", desc_df["StockCode"].astype(str).values)
    out.to_parquet(cache, index=False)
    return out


if __name__ == "__main__":
    from .descriptions import make_name_descriptions, make_llm_descriptions
    backend = _resolve_backend(C.EMBEDDING_BACKEND)
    print(f"backend = {backend}, dim = {C.EMBEDDING_DIM}")
    name = embed_descriptions(make_name_descriptions(), "name", force=True)
    print("name embeddings:", name.shape)
    try:
        llm = embed_descriptions(make_llm_descriptions(), "llm", force=True)
        print("llm embeddings:", llm.shape)
    except FileNotFoundError:
        print("llm descriptions not generated yet — run descriptions.py first")
