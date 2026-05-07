import ollama
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re
# Load the dataset
def load_and_chunk(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 50:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# --- Quality Filter ---
def is_high_quality(chunk):
    # Length filter
    if len(chunk) < 80:
        return False

    words = re.findall(r'\w+', chunk.lower())

    # Too few words
    if len(words) < 10:
        return False

    # Lexical diversity
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.4:
        return False

    # Ensure it's not a fragment (must contain sentence structure)
    if "." not in chunk:
        return False

    return True


# --- Deduplication ---
def deduplicate(chunks):
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        key = chunk[:200]  # simple prefix-based dedupe
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks


# --- Load + Process ---
with open('docs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize whitespace
text = re.sub(r'\s+', ' ', text)

# Step 1: Chunk
dataset = load_and_chunk(text)
print(f"Loaded {len(dataset)} chunks")

# Step 2: Filter
filtered_dataset = [c for c in dataset if is_high_quality(c)]
print(f"After filtering: {len(filtered_dataset)} chunks")

# Step 3: Deduplicate
dataset = deduplicate(filtered_dataset)
print(f"After deduplication: {len(dataset)} chunks")


# --- Sanity Check ---
for i in range(min(3, len(dataset))):
    print(f"\n--- Chunk {i} ---\n{dataset[i]}")
# Models
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
# --- Storage ---
CHUNKS = []
FAISS_INDEX = None
# --- BM25 Setup ---
# BM25 works on tokenized text (list of words), not raw strings
def tokenize(text):
    return re.findall(r'\w+', text.lower())  # lowercase + split on words
TOKENIZED_CORPUS = []  # list of tokenized chunks, used to build BM25
BM25 = None            # will be rebuilt after all chunks are added
# --- Indexing ---
def add_chunk_to_database(chunk):
    global FAISS_INDEX
    # Vector side
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    embedding_np = np.array(embedding, dtype=np.float32)
    if FAISS_INDEX is None:
        dim = len(embedding_np)
        FAISS_INDEX = faiss.IndexFlatL2(dim)
    CHUNKS.append(chunk)
    FAISS_INDEX.add(embedding_np.reshape(1, -1))
    # BM25 side
    TOKENIZED_CORPUS.append(tokenize(chunk))
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')
# Build BM25 index AFTER all chunks are added
BM25 = BM25Okapi(TOKENIZED_CORPUS)
print('BM25 index built.')
# --- Hybrid Retrieval ---
def retrieve(query, top_n=5, alpha=0.6):
    """
    alpha controls the blend:
      alpha=1.0 → pure vector search
      alpha=0.0 → pure BM25
      alpha=0.5 → equal blend (default)
    """
    num_chunks = len(CHUNKS)
    # --- Vector scores ---
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    # Fetch all vectors so we can score every chunk (not just top_n)
    distances, indices = FAISS_INDEX.search(query_np, num_chunks)
    vector_scores = np.zeros(num_chunks)
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            vector_scores[idx] = 1 / (1 + dist)  # L2 distance → similarity
    # --- BM25 scores ---
    bm25_raw = BM25.get_scores(tokenize(query))  # raw scores, one per chunk
    # --- Normalize both score arrays to [0, 1] so they're on the same scale ---
    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    vector_scores_norm = normalize(vector_scores)
    bm25_scores_norm   = normalize(np.array(bm25_raw))
    # --- Combine ---
    hybrid_scores = alpha * vector_scores_norm + (1 - alpha) * bm25_scores_norm
    # Sort by hybrid score descending
    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    return [(CHUNKS[i], hybrid_scores[i]) for i in ranked_indices]
# --- Chatbot ---
while True:
    input_query = input('Ask me a question: ').strip()
    if input_query == "quit":
        break
    retrieved_knowledge = retrieve(input_query)
    print('Retrieved knowledge:')
    for chunk, score in retrieved_knowledge:
        print(f' - (score: {score:.2f}) {chunk}')
    context_str = "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    instruction_prompt = f"""You are a helpful assistant.

    Answer the question using ONLY the provided context.

    Guidelines:
    - Combine information from multiple context chunks
    - Provide a clear and complete explanation
    - Include key supporting details if available
    - Keep the answer concise but informative

    Do NOT:
    - Use external knowledge
    - Make unsupported assumptions

    If the answer cannot be reasonably inferred, say "I don't know."

    CONTEXT:
    {context_str}
    """
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )
    print('\nChatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")
