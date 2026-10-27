# Program 25: Sentence embeddings using BERT and GPT-2
# Colab: !pip install -q transformers torch
import torch
from transformers import AutoTokenizer, AutoModel
sentences = ["NLP studies human language.", "NLP enables computers to process text."]
def pool(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
for name in ["bert-base-uncased", "gpt2"]:
    print("\nModel:", name)
    tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    encoded = tok(sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad(): out = model(**encoded)
    emb = torch.nn.functional.normalize(pool(out.last_hidden_state, encoded["attention_mask"]), p=2, dim=1)
    print("Embedding shape:", tuple(emb.shape))
    print("Cosine similarity:", round(float((emb[0] * emb[1]).sum()), 3))
