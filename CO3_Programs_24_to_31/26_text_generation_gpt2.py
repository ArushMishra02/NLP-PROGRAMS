# Program 26: Text generation using GPT-2
# Colab: !pip install -q transformers torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name)
prompt = "Natural language processing allows computers to"
inputs = tok(prompt, return_tensors="pt")
torch.manual_seed(42)
out = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=.8, top_p=.9, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
