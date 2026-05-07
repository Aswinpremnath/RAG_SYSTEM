"""
Optimized Local RAG Pipeline — LangChain + LangSmith Edition
=============================================================
Rewritten from the 2026 Edition to use LangChain abstractions and
LangSmith tracing. Only what was necessary has changed; all logic
(RRF, semantic chunking, quality filters, dedup, reranking,
context compression, streaming) is preserved verbatim or near-verbatim.

Requirements:
    pip install langchain langchain-community langchain-huggingface \
                faiss-cpu bm25s sentence-transformers fastembed \
                langsmith ollama numpy
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=<your-langsmith-key>
    export LANGCHAIN_PROJECT=local-rag          # optional project name
"""

import os
import re
import json
import pickle
import hashlib
import numpy as np
import faiss
import bm25s

from pathlib import Path

# ── LangChain imports (replaces direct ollama / fastembed calls) ──────────────
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# ── LangSmith (tracing only — zero code change to logic) ─────────────────────
from langsmith import traceable          # decorator: wraps any function as a traced span
from langsmith.wrappers import wrap_openai  # NOT used here, imported for reference only

# ── sentence-transformers (reranker — unchanged) ──────────────────────────────
from sentence_transformers import CrossEncoder

# ─── Config ───────────────────────────────────────────────────────────────────
#  All values identical to the original.

INDEX_DIR        = Path("rag_index")
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LANGUAGE_MODEL   = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 80
RETRIEVE_TOP_N   = 20
RERANK_TOP_N     = 5
RRF_K            = 60
MIN_CHUNK_CHARS  = 80
MIN_CHUNK_WORDS  = 10

# ─── Model loading ────────────────────────────────────────────────────────────

print("Loading embedding model …")
EMBEDDER = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Loading LLM …")
LLM = ChatOllama(model=LANGUAGE_MODEL)

print("Loading reranker …")
RERANKER = CrossEncoder(RERANKER_MODEL)   # UNCHANGED — no LC wrapper for cross-encoders


# ─── Semantic chunker ─────────────────────────────────────────────────────────

def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]
    chunks, start_idx = [], 0

    while start_idx < len(sentences):
        current, current_len, idx = [], 0, start_idx
        while idx < len(sentences) and current_len + len(sentences[idx]) <= chunk_size:
            current.append(sentences[idx])
            current_len += len(sentences[idx]) + 1
            idx += 1
        if not current and idx < len(sentences):
            current.append(sentences[idx])
            idx += 1
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        overlap_chars, new_start = 0, idx
        for i in range(idx - 1, start_idx - 1, -1):
            overlap_chars += len(sentences[i])
            if overlap_chars >= overlap:
                new_start = i
                break
        start_idx = max(new_start, start_idx + 1)

    return chunks


# ─── Quality filter ───────────────────────────────────────────────────────────

def is_high_quality(chunk: str) -> bool:
    if len(chunk) < MIN_CHUNK_CHARS:
        return False
    words = re.findall(r'\w+', chunk.lower())
    if len(words) < MIN_CHUNK_WORDS:
        return False
    if len(set(words)) / len(words) < 0.35:
        return False
    if not re.search(r'[.!?]', chunk):
        return False
    return True


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(chunks: list[str]) -> list[str]:
    seen, unique = set(), []
    for chunk in chunks:
        key = hashlib.md5(chunk[:200].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


# ─── Persistent index ─────────────────────────────────────────────────────────

def save_index(chunks: list[str], faiss_index, bm25_index) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_DIR / "vectors.faiss"))
    bm25_index.save(str(INDEX_DIR / "bm25"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {INDEX_DIR}/")


def load_index():
    if not (INDEX_DIR / "vectors.faiss").exists():
        return None
    print("Loading existing index from disk …")
    faiss_index = faiss.read_index(str(INDEX_DIR / "vectors.faiss"))
    bm25_index = bm25s.BM25.load(str(INDEX_DIR / "bm25"), load_corpus=True)
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from disk.")
    return chunks, faiss_index, bm25_index


# ─── Index builder ────────────────────────────────────────────────────────────

def build_index(chunks: list[str]):
    print(f"Embedding {len(chunks)} chunks …")

  
    embeddings = EMBEDDER.embed_documents(chunks)          # List[List[float]]
    matrix = np.array(embeddings, dtype=np.float32)        # (N, 768) — identical shape
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 64
    faiss_index.add(matrix)

    print("Building BM25S index …")
    tokenized = bm25s.tokenize([c for c in chunks])
    bm25_index = bm25s.BM25()
    bm25_index.index(tokenized)

    return faiss_index, bm25_index


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever(BaseRetriever):
    """
    Wraps the original two-stage hybrid retrieval (vector + BM25 → RRF →
    cross-encoder reranking) as a LangChain BaseRetriever so it integrates
    with LCEL chains and LangSmith tracing.
    """
    chunks: list
    faiss_index: object
    bm25_index: object
    top_n: int = RETRIEVE_TOP_N
    rerank_top_n: int = RERANK_TOP_N

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # ── Stage 1a: Vector search ──────────────────────────────────────────
        # CHANGED: embed_query instead of embed([query])
        query_vec = np.array(
            [EMBEDDER.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_vec)
        _, vector_indices = self.faiss_index.search(query_vec, self.top_n)
        vector_ranking = [int(i) for i in vector_indices[0] if i != -1]

        # ── Stage 1b: BM25 search ────────────────────────────────────────────
        query_tokens = bm25s.tokenize([query])
        bm25_results, _ = self.bm25_index.retrieve(query_tokens, k=self.top_n)
        bm25_ranking = [int(i) for i in bm25_results[0]]

        # ── Stage 1c: RRF fusion ─────────────────────────────────────────────
        fused_ranking = reciprocal_rank_fusion(
            [vector_ranking, bm25_ranking]
        )[:self.top_n]
        candidates = [self.chunks[i] for i in fused_ranking]

        # ── Stage 2: Cross-encoder reranking ─────────────────────────────────
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = RERANKER.predict(pairs)
        scored = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )
        top = scored[:self.rerank_top_n]

        # CHANGED: return List[Document] instead of List[Tuple[str, float]]
        return [
            Document(page_content=chunk, metadata={"rerank_score": float(score)})
            for chunk, score in top
        ]


# ─── Context compressor ───────────────────────────────────────────────────────

@traceable(name="compress_context")
def compress_context(query: str, docs: list[Document]) -> str:
    compressed = []
    for doc in docs:
        chunk = doc.page_content          # CHANGED: unwrap from Document
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        if len(sentences) <= 2:
            compressed.append(chunk)
            continue
        pairs = [[query, s] for s in sentences]
        scores = RERANKER.predict(pairs)
        threshold = float(np.median(scores))
        kept = [s for s, sc in zip(sentences, scores) if sc >= threshold]
        compressed.append(" ".join(kept) if kept else chunk)

    return "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(compressed)])


# ─── LLM query rewriter ───────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a retrieval query optimizer. "
        "Rewrite the user's question into a clear, detailed query "
        "that will retrieve the most relevant passages from a document store. "
        "Output only the rewritten query — no preamble, no explanation."
    ),
    HumanMessagePromptTemplate.from_template("{question}"),
])

# LCEL chain: prompt → LLM → parse to string
_rewrite_chain = _REWRITE_PROMPT | LLM | StrOutputParser()

@traceable(name="rewrite_query")
def rewrite_query(raw_query: str) -> str:
    return _rewrite_chain.invoke({"question": raw_query})


# ─── Generation ───────────────────────────────────────────────────────────────


_ANSWER_SYSTEM = """\
You are a precise, helpful assistant.

Answer the user's question using ONLY the numbered context passages below.

Rules:
- Cite the passage number(s) you draw from, e.g. [1], [2].
- Combine information from multiple passages when relevant.
- Be concise and factual.
- If the answer is not in the context, say "I don't know based on the provided documents."
- Never use outside knowledge.

CONTEXT:
{context}
"""

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(_ANSWER_SYSTEM),
    HumanMessagePromptTemplate.from_template("{question}"),
])

@traceable(name="generate_answer")
def generate_answer(query: str, context: str) -> None:
 
    stream = (_ANSWER_PROMPT | LLM).stream(
        {"context": context, "question": query}
    )
    print("\nAnswer:")
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print("\n")


def build_rag_chain(retriever: HybridRetriever):
    """
    Composes the full RAG pipeline as an LCEL chain.
    Inputs:  {"question": str}
    Outputs: streamed answer printed to stdout; returns context str.
    """
    def _retrieve_and_compress(inputs: dict) -> dict:
        query = inputs["question"]
        docs = retriever.get_relevant_documents(query)
        context = compress_context(query, docs)
        return {"question": query, "context": context}

    chain = (
        RunnableLambda(_retrieve_and_compress)
        | RunnablePassthrough.assign(
            answer=RunnableLambda(
                lambda x: generate_answer(x["question"], x["context"])
            )
        )
    )
    return chain


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Try loading an existing index ─────────────────────────────────────
    loaded = load_index()

    if loaded is not None:
        chunks, faiss_index, bm25_index = loaded
    else:
        print("Building index from docs.txt …")
        with open("docs.txt", "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"\s+", " ", text).strip()

        raw_chunks   = semantic_chunk(text)
        print(f"Raw chunks:       {len(raw_chunks)}")
        filtered     = [c for c in raw_chunks if is_high_quality(c)]
        print(f"After filtering:  {len(filtered)}")
        chunks       = deduplicate(filtered)
        print(f"After dedup:      {len(chunks)}")

        faiss_index, bm25_index = build_index(chunks)
        save_index(chunks, faiss_index, bm25_index)

    # ── 2. Build retriever ────────────────────────────────────────────────────
    
    retriever = HybridRetriever(
        chunks=chunks,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
    )

    # ── 3. Chat loop ──────────────────────────────────────────────────────────
    print("\nRAG chatbot ready. Type 'quit' to exit.\n")

    while True:
        raw_query = input("You: ").strip()
        if raw_query.lower() in ("quit", "exit", "q"):
            break
        if not raw_query:
            continue

        # Step A: rewrite
        print("  [rewriting query …]")
        retrieval_query = rewrite_query(raw_query)
        print(f"  Rewritten: {retrieval_query}")

        # Step B: retrieve via HybridRetriever
        print(f"  [retrieving top {RETRIEVE_TOP_N} → reranking to {RERANK_TOP_N} …]")
        docs = retriever.invoke(retrieval_query)

        if not docs:
            print("No relevant passages found.\n")
            continue

        # Step C: compress context
        context = compress_context(retrieval_query, docs)

        # Step D: stream answer
        generate_answer(raw_query, context)


if __name__ == "__main__":
    main()