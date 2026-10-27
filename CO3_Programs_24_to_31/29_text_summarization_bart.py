# Program 29: Text summarization using BART
# Colab: !pip install -q transformers torch
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
text = """Natural language processing is a branch of artificial intelligence that enables computers to analyze, understand, and generate human language. Common tasks include tokenization, sentiment analysis, translation, summarization, information retrieval, and question answering. Modern systems often use neural networks and transformer architectures. Quality depends on data, model design, evaluation, and context."""
print(summarizer(text, max_length=55, min_length=20, do_sample=False)[0]["summary_text"])
