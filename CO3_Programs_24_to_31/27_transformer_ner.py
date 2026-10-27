# Program 27: Named Entity Recognition using Transformers
# Colab: !pip install -q transformers torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
name = "dslim/bert-base-NER"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForTokenClassification.from_pretrained(name)
ner = pipeline("ner", model=model, tokenizer=tok, aggregation_strategy="simple", device=-1)
text = "Satya Nadella works at Microsoft in Redmond and visited London."
print("Text:", text)
for e in ner(text): print(f"{e['word']} -> {e['entity_group']} ({e['score']:.3f})")
