# Program 23: Implement a basic custom attention mechanism
import numpy as np
tokens = ["The", "student", "learns", "NLP"]
embeddings = np.array([[.2,.1,.7],[.8,.3,.2],[.4,.9,.1],[.6,.2,.8]])
query = np.array([.5,.7,.2])
scores = embeddings @ query
exp_scores = np.exp(scores - scores.max())
weights = exp_scores / exp_scores.sum()
context = weights @ embeddings
for token, score, weight in zip(tokens, scores, weights):
    print(f"{token:10} score={score:.3f} attention={weight:.3f}")
print("Context vector:", context.round(3))
