from google import genai 
import math

client = genai.Client()

def get_dense_embeddings(text) -> list[float]: 
    # 768-dim embeddings due to google's embedding model
    response = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = text,
    )

    return response.embeddings[0].values

def calculate_cosine_similarity(a,b):
    n = sum(x*y for x,y in zip(a,b))
    d1 = math.sqrt(sum(x**2 for x in a))
    d2 = math.sqrt(sum(y**2 for y in b)) 
    if d1 == 0 or d2 == 0:
          return 0.0
    return n/(d1*d2)

def retrieve_top_k(query,chunks,chunk_vectors,top_k):
        query_vector = get_dense_embeddings(query)
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

def call_llm (prompt):
    response = client.models.generate_content(
        model = "gemini-3.6-flash",
        contents = prompt,
    )
    return response.text

SAMPLE_CHUNKS = [
    "Project Apollo was created in 1961 by NASA to land humans on the Moon.",
    "The Saturn V rocket was the launch vehicle developed under the direction of Wernher von Braun.",
    "On July 20, 1969, Apollo 11 successfully touched down in the Sea of Tranquility.",
    "Astronaut Neil Armstrong became the first human to step onto the lunar surface.",
    "The command module was named Columbia, and the lunar module was named Eagle."
]


if __name__ == "__main__":

    vector_chunks = [get_dense_embeddings(chunk) for chunk in SAMPLE_CHUNKS]
    query = "What were the names of the Apollo 11 modules?"
    top_k = 3
    top_matches = retrieve_top_k(query,SAMPLE_CHUNKS,vector_chunks,top_k)

    for rank, (score, text) in enumerate(top_matches):
        print(f"   [Rank {rank+1} | Score: {score:.4f}] -> \"{text}\"")

    # Augment and generate
    augmented_prompt = generate_augmented_prompt(query,top_matches)
    print(f'augmented prompt fed to the model:')
    print(augmented_prompt)

    # llm answer
    response = call_llm(augmented_prompt)
    print(f'LLM response: \n {response}')
