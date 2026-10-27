# Program 28: Machine translation using MarianMT
# Colab: !pip install -q transformers sentencepiece torch
from transformers import MarianMTModel, MarianTokenizer
name = "Helsinki-NLP/opus-mt-en-hi"
tok = MarianTokenizer.from_pretrained(name); model = MarianMTModel.from_pretrained(name)
sentences = ["Natural language processing is useful.", "Artificial intelligence can help students learn."]
inputs = tok(sentences, return_tensors="pt", padding=True, truncation=True)
out = model.generate(**inputs, max_new_tokens=80)
for src, pred in zip(sentences, out):
    print("English:", src)
    print("Hindi:", tok.decode(pred, skip_special_tokens=True), "\n")
