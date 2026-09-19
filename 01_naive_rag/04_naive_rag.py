import re
import os
import math
from collections import Counter
import json
import urllib.request


def split_chunks_with_overlap(text,chunk_size,overlap_size):
    chunk = []
    if overlap_size>=chunk_size:
        raise ValueError("Overlap should be smaller then chunk size")
    for i in range(0,len(text), chunk_size-overlap_size):    
        chunk.append(text[i:i+chunk_size])
    return chunk

def tokenize(chunk) -> list[str]:
        chunk = chunk.lower()
        return re.findall(r'\b[a-z0-9]+\b',chunk)

def build_vocabulary_and_idf(tokenchunks):
        vocabulary = set()
        for t in tokenchunks:
            vocabulary.update(t)

        idf = {}
        for word in vocabulary:
            doc_count = 0
            for chunk in tokenchunks:
                if word in chunk:
                        doc_count += 1

            idf[word] = math.log((len(tokenchunks)+1)/(doc_count+1))+1.0
        vocabulary = sorted(vocabulary)
        return vocabulary,idf


def build_vector(tokenized_chunk,vocab,idf):
        embedding = []
        word_counts = Counter(tokenized_chunk)
        for word in vocab:
            token_tf = word_counts[word]/max(len(tokenized_chunk),1)
            embedding.append(token_tf*idf[word])
        return embedding

def calculate_cosine_similarity(a,b):
    n = sum(x*y for x,y in zip(a,b))
    d1 = math.sqrt(sum(x**2 for x in a))
    d2 = math.sqrt(sum(y**2 for y in b)) 
    if d1 == 0 or d2 == 0:
          return 0.0
    
    return n/(d1*d2)

def retrieve_top_k(query,chunks,chunk_vectors,vocab,idf,top_k):
        query_tokens = tokenize(query)
        query_vector = build_vector(query_tokens,vocab,idf)
        scored_chunks  = []
        for chunk_text, chunk_vec in zip(chunks,chunk_vectors):
            scored_chunks.append([calculate_cosine_similarity(query_vector,chunk_vec),chunk_text])

        scored_chunks.sort(key = lambda item: item[0], reverse=True)

        return scored_chunks[:top_k]

def generate_augmented_prompt(user_query,top_matches):
    context_text = "\n\n".join(
          f'Context Chunk {i+1} . Relevance: {score} . text: \n {text}'
          for i,(score,text) in enumerate(top_matches)
     )
    prompt = f'''You are a good ai. Use only the text give in the context below. Don't make stuff up and if there is no info in the context about the query say the 'I could not find the information in the provided documents
     
    ===Context start===
    {context_text}
    ===Context end===

    User question: {user_query}
    Answer:'''
    return prompt    

def call_llm(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return  'issueNOTE: GEMINI_API_KEY is not set in your terminal yet'
        
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"    
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req  = urllib.request.Request(url,data=payload,headers=headers,method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f'Error with gemini api: {e}'



SAMPLE_DOC = (
    "Project Apollo was created in 1961 by NASA to land humans on the Moon. "
    "The Saturn V rocket was the launch vehicle developed under the direction of Wernher von Braun. "
    "On July 20, 1969, Apollo 11 successfully touched down in the Sea of Tranquility. "
    "Astronaut Neil Armstrong became the first human to step onto the lunar surface, "
    "uttering the famous words: 'That's one small step for man, one giant leap for mankind.' "
    "The command module was named Columbia, and the lunar module was named Eagle."
)

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 RUNNING FULL NAIVE RAG PIPELINE")
    print("=" * 65)

    # chunking
    chunks = split_chunks_with_overlap(SAMPLE_DOC,chunk_size=140,overlap_size=35)
    print(f'Document ingested and {len(chunks)} no of overlapping chunks created')

    # Vectorizing
    chunk_tokens = [tokenize(chunk) for chunk in chunks]
    vocab,idf = build_vocabulary_and_idf(chunk_tokens)
    chunk_vectors = [build_vector(tokenized_chunk,vocab,idf) for tokenized_chunk in chunk_tokens]
    print(f"chunk vectors created")

    # retrieve
    user_query = "What were the names of the Apollo 11 modules?"
    print(f'User query: {user_query}')
    top_k = 3
    print(f'Gettign top {top_k} chunks')
    top_matches = retrieve_top_k(user_query,chunks,chunk_vectors,vocab,idf,top_k)
    for rank, (score, text) in enumerate(top_matches):
        print(f"   [Rank {rank+1} | Score: {score:.4f}] -> \"{text}\"")

    # Augment and generate
    augmented_prompt = generate_augmented_prompt(user_query,top_matches)
    print(f'augmented prompt fed to the model:')
    print(augmented_prompt)

    # llm answer
    response = call_llm(augmented_prompt)
    print(f'LLM response: \n {response}')
