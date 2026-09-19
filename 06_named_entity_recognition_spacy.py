# Program 6: Named Entity Recognition (NER) using spaCy

# Install once in Google Colab:
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Arush is studying Computer Science at NIET in Greater Noida.
He visited New Delhi in August and spent Rs. 500 at a cafe.
"""

doc = nlp(text)

print("========== NAMED ENTITY RECOGNITION ==========")
print("Text:")
print(text)

print("\nEntities:")
if not doc.ents:
    print("No named entities found.")
else:
    for entity in doc.ents:
        print(f"Text: {entity.text:20} | Label: {entity.label_:10} | Explanation: {spacy.explain(entity.label_)}")
