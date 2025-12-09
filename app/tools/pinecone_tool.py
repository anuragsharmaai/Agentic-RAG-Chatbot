import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        # Primary embedding: SentenceTransformers (BERT)
        self.embed_dim = 768
        self.use_sbert = True
        try:
            self.sbert_model = SentenceTransformer(
                "sentence-transformers/bert-base-nli-mean-tokens",
                trust_remote_code=False,
            )
        except Exception:
            self.use_sbert = False
            self.sbert_model = None

        # Secondary embedding: Google Generative AI
        self.embed_model = "models/text-embedding-004"
        self.use_google = bool(settings.google_api_key)
        if self.use_google:
            try:
                genai.configure(api_key=settings.google_api_key)
            except Exception:
                self.use_google = False
        self.client = None
        self.index = None
        self.index_name = settings.pinecone_index
        try:
            self.client = Pinecone(api_key=settings.pinecone_api_key)
            self._ensure_index()
            self.index = self.client.Index(self.index_name)
        except Exception:
            self.client = None
            self.index = None

    def _ensure_index(self):
        existing = []
        for idx in self.client.list_indexes():
            existing.append(idx.name)
        if self.index_name not in existing:
            try:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.embed_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env)
                )
            except Exception:
                self.index = None
            self.client.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env)
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Prefer SentenceTransformers if available
        if self.use_sbert and self.sbert_model is not None:
            try:
                vecs = self.sbert_model.encode(texts)
                # vecs is a numpy array (n, 768)
                return [v.tolist() for v in vecs]
            except Exception:
                raise RuntimeError("SentenceTransformers embedding failed")

        # Fallback to Google embeddings when configured
        if self.use_google:
            vectors: List[List[float]] = []
            for text in texts:
                try:
                    res = genai.embed_content(model=self.embed_model, content=text)
                    emb = res.get("embedding") or res.get("data", {}).get("embedding")
                    if isinstance(emb, dict):
                        emb = emb.get("values")
                    if not emb or not isinstance(emb, list):
                        raise RuntimeError("Google embedding returned invalid vector")
                    vectors.append([float(x) for x in emb])
                except Exception as e:
                    raise RuntimeError(f"Google embedding failed: {e}")
            return vectors

        # No embedding provider available
        raise RuntimeError("No embedding provider available. Configure SentenceTransformers or GOOGLE_API_KEY.")


    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        if chunk_size <= 0:
            return [text]
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 4)
        chunks: List[str] = []
        n = len(text or "")
        if n == 0:
            return []
        start = 0
        while start < n:
            end = min(n, start + chunk_size)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def upsert_documents(self, docs: List[Dict[str, Any]]):
        # Chunk each document, embed chunks, upsert per-chunk
        items: List[Dict[str, Any]] = []
        for d in docs:
            doc_id = d["id"]
            raw_text = d.get("text", "")
            base_meta = d.get("metadata", {}) or {}
            # Use the default chunking parameters configured in _chunk_text
            chunks = self._chunk_text(raw_text)
            if not chunks:
                continue
            vectors = self.embed_texts(chunks)
            for idx, vec in enumerate(vectors):
                cid = f"{doc_id}::{idx}"
                meta = dict(base_meta)
                meta["source_id"] = doc_id
                meta["chunk"] = idx
                # Persist the actual chunk text so it can be retrieved as RAG context
                meta["text"] = chunks[idx]
                item = {
                    "id": cid,
                    "values": vec,
                    "metadata": meta,
                }
                items.append(item)
        if self.index is None:
            return
        try:
            self.index.upsert(vectors=items)
        except Exception:
            return

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qvecs = self.embed_texts([query])
        qvec = qvecs[0]
        if self.index is None:
            return []
        try:
            res = self.index.query(vector=qvec, top_k=k, include_metadata=True)
        except Exception:
            return []
        results = []
        for match in res.matches:
            record = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            results.append(record)
            # print(results)
        return results

vector_store = VectorStore()
