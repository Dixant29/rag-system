import re
import math
from collections import Counter

def tokenize(chunk) -> list[str]:
        # for char in [",",".","?","!","-"]:
        #     chunk = chunk.replace(char," ")
       
        # return chunk.lower().split()
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

if __name__ == "__main__":
        chunks = [
        "In early applications, computers relied strictly on hand-crafted rules.",
        "Today, Deep Learning enables systems to learn directly from massive datasets.",
        "Retrieval-Augmented Generation bridges static models with external private documents."
    ]
        
        tokenchunks = [tokenize(c) for c in chunks]

        vocab,idf = build_vocabulary_and_idf(tokenchunks)
        print(vocab)
        embeddings = [build_vector(tokenized_chunk,vocab,idf) for tokenized_chunk in tokenchunks]


        query = "How does deep learning work?"
        query_tokens = tokenize(query)
        query_embeddings = build_vector(query_tokens,vocab,idf)


        scores = [calculate_cosine_similarity(query_embeddings,embedding) for embedding in embeddings]


        ranked_indices = sorted(range(len(scores)),key = lambda i: scores[i],reverse= True)

        for i,idx in enumerate(ranked_indices):
              print(f'Ranked: {i+1} [ Score: {scores[idx]}]: \" {chunks[idx]}\"')