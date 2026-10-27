# Program 30: Chatbot using pretrained DialoGPT
# Colab: !pip install -q transformers torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
name = "microsoft/DialoGPT-small"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name)
history = None
print("Type 'quit' to stop.")
while True:
    text = input("You: ").strip()
    if text.lower() in {"quit", "exit"}:
        print("Bot: Goodbye!"); break
    new = tok.encode(text + tok.eos_token, return_tensors="pt")
    context = torch.cat([history, new], dim=-1) if history is not None else new
    history = model.generate(context, max_length=min(context.shape[-1] + 60, 900), pad_token_id=tok.eos_token_id, do_sample=True, top_p=.92, temperature=.75)
    print("Bot:", tok.decode(history[:, context.shape[-1]:][0], skip_special_tokens=True))
