# Program 24: Fine-tuning BERT for text classification using Hugging Face
# Colab: !pip install -q transformers datasets accelerate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
model_name = "distilbert-base-uncased"
texts = ["I love this product","wonderful experience","service is excellent","very happy with result","fantastic quality","I hate this product","terrible experience","service is awful","disappointed with result","poor quality"]
labels = [1]*5 + [0]*5
ds = Dataset.from_dict({"text": texts, "label": labels}).train_test_split(test_size=.2, seed=42)
tok = AutoTokenizer.from_pretrained(model_name)
def encode(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=64)
ds = ds.map(encode, batched=True).remove_columns(["text"])
ds.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
args = TrainingArguments(output_dir="./bert-result", eval_strategy="epoch", save_strategy="no", num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=4, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
print(trainer.evaluate())
text = "The product is wonderful and I am happy"
inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
print(text, "->", ["Negative", "Positive"][int(model(**inputs).logits.argmax(-1))])
