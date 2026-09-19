# RAG System: From Scratch to Advanced

A pedagogical, step-by-step implementation of Retrieval-Augmented Generation (RAG) built from first principles in Python.

---

## 🚀 Phase 1: Naive RAG (From Scratch)

This phase explores the internal mechanics of retrieval systems without relying on high-level frameworks like LangChain or LlamaIndex.

### Modules (Folder: `01_naive_rag/`)

1. **`01_similarity.py`**
   * Intuition behind embeddings & vector dimensions.
   * Pure-Python implementation of Cosine Similarity ($\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$) using dot products and Euclidean norms.

2. **`02_chunking.py`**
   * Why chunking matters (avoiding information dilution and model limits).
   * Sliding window text chunking with character/token overlap to preserve semantic continuity.

3. **`03_tfidf_vectors.py`**
   * Tokenization and vocabulary building.
   * Full TF-IDF (Term Frequency - Inverse Document Frequency) implementation from scratch.
   * Generating document vectors and measuring sparse similarity.

4. **`04_naive_rag.py`**
   * End-to-end Naive RAG pipeline:
     $$\text{Raw Document} \to \text{Chunking} \to \text{Vector Ingestion} \to \text{Query Search} \to \text{Prompt Assembly} \to \text{Generation}$$
   * Live Gemini model generation and grounding validation.

---

## ⚡ Phase 2: Advanced Retrieval (Underway)

This phase upgrades the retrieval engine from lexical (word count) matching to semantic and hybrid retrieval.

### Modules (Folder: `02_advanced_retrieval/`)

1. **`01_dense_embeddings.py`**
   * Dense Neural Semantic Embeddings using Google's `text-embedding-004` (768 dimensions).
   * Solving the Out-of-Vocabulary (OOV) and semantic synonym gap that broke Naive RAG.

---

## 🛣️ Roadmap

- [x] **Phase 1: Naive RAG** (Pure Python foundations)
- [ ] **Phase 2: Advanced Retrieval** (Dense Embeddings, Multi-Query, Hybrid BM25, Cross-Encoder Re-ranking)
- [ ] **Phase 3: Agentic RAG** (Self-reflective loops, Query rewriting, Tool use)

