# Program 20: Information extraction from structured/unstructured documents
# Colab: !python -m spacy download en_core_web_sm
import re, spacy
nlp = spacy.load("en_core_web_sm")
text = """On 14 March 2026, NovaTech announced that it hired 25 engineers in New Delhi.
Contact hiring@novatech.example or call +91 98765 43210. The budget is Rs. 12,50,000."""
doc = nlp(text)
print("Named entities:")
for e in doc.ents: print(f"{e.text} -> {e.label_}")
print("Emails:", re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text))
print("Phone numbers:", re.findall(r"\+?\d[\d\s-]{8,}\d", text))
print("Budget:", re.findall(r"(?:Rs\.?|INR|₹)\s?[\d,]+", text))
