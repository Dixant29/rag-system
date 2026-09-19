def split_chunks_with_overlap(text,chunk_size,overlap_size):
    chunks = []
    if overlap_size >= chunk_size:
        raise ValueError("Overlap should be smaller than chunk size")
    for i in range(0,len(text), chunk_size-overlap_size):    
        chunks.append(text[i:i+chunk_size])
    return chunks


SAMPLE_DOCUMENT = (
    "Artificial Intelligence has evolved dramatically over the last decade. "
    "In early applications, computers relied strictly on hand-crafted rules. "
    "Today, Deep Learning enables systems to learn directly from massive datasets. "
    "Retrieval-Augmented Generation (RAG) is a modern architecture that bridges "
    "static models with dynamic, external knowledge sources like private documents."
)

if __name__ == "__main__":
    chunks = split_chunks_with_overlap(SAMPLE_DOCUMENT, 50,10)
    for i, chunk in enumerate(chunks):
        print(f'\n[Chunk {i}] (Length: {len(chunk)}):')
        print(f'"{chunk}"')