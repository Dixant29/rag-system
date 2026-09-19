
import math

def cosine_similarity(x,y):
    n = 0
    d1 = 0
    d2 = 0

    n = sum(m*n for m,n in zip(x,y))

    d1 = sum(x[i]**2 for i in range(len(x)))
    d2 = sum(y[i]**2 for i in range(len(y)))
    if d1 == 0 or d2 == 0:
        return 0.0
    return (n/(math.sqrt(d1)*math.sqrt(d2)))

if __name__ == "__main__":
    A = [0.9,0.9,0]
    B = [0.9,0.6,0]
    score = cosine_similarity(A,B)
    print(f"score is : {score:.4f}")