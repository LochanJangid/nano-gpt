import json
import torch
from model import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load our itos, stoi data first 
tokenizers = json.load(open("tokenizer.json", "r"))
itos = {int(k): v for k, v in tokenizers["itos"].items()} # json save keys in string format
stoi = tokenizers["stoi"]
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long, device=device)
decode = lambda x: "".join([itos[token.item()] for token in x])

# Get Trained Model
m = GPT(vocab_size)

m.load_state_dict(torch.load("nano_gpt.pth", map_location=device, weights_only=True))

m = m.to(device)

start = encode("Ram").unsqueeze(dim=0)

generated_tokens = m.generate(start, max_new_tokens=500)[0]

output = decode(generated_tokens)

print(output)