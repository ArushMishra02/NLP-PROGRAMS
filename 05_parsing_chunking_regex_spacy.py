# Program 5: Parsing and Chunking using RegEx and spaCy

# Install once in Google Colab:
# !pip install nltk spacy
# !python -m spacy download en_core_web_sm

import re
import nltk
import spacy
from nltk import word_tokenize, pos_tag

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

sentence = "The smart student solved a difficult problem."

# =========================================================
# PART A: Chunking using Regular Expression
# =========================================================
print("========== REGEX CHUNKING ==========")

words = word_tokenize(sentence)
tagged_words = pos_tag(words)

# NP pattern: Determiner + optional adjective(s) + noun
grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
"""

chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(tagged_words)

print(tree)

print("\nNoun Phrases found using Regex:")
for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
    print(" ".join(word for word, tag in subtree.leaves()))

# =========================================================
# PART B: Parsing using spaCy Dependency Parser
# =========================================================
print("\n========== spaCy DEPENDENCY PARSING ==========")

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)

print("Token | Dependency | Head")
print("-" * 35)

for token in doc:
    print(f"{token.text:10} | {token.dep_:12} | {token.head.text}")

print("\nNoun chunks using spaCy:")
for chunk in doc.noun_chunks:
    print(chunk.text)
