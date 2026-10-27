# Program 31: Automatic document classification and separation using ML
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
texts = ["football match player team goal", "cricket batsman wicket tournament", "basketball player scored points", "python software database server", "computer hardware programming application", "machine learning algorithm data model", "doctor hospital patient treatment", "nurse clinic health medical care", "diagnosis medicine hospital doctor"]
labels = ["Sports"]*3 + ["Technology"]*3 + ["Healthcare"]*3
model = Pipeline([("tfidf", TfidfVectorizer()), ("svm", LinearSVC())]).fit(texts, labels)
docs = {"doc_01.txt":"The cricket team won the tournament.", "doc_02.txt":"The application uses Python and a database.", "doc_03.txt":"The doctor discussed treatment at the hospital."}
root = Path("classified_documents")
for filename, content in docs.items():
    category = model.predict([content])[0]
    folder = root / category; folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding="utf-8")
    print(f"{filename} -> {category}/{filename}")
print("Output folder:", root.resolve())
