import json
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT

# Hyperparameters ----------------
context_size = 256
batch_size = 64 
eval_iter = 200 
max_iter = 5000 
eval_interval = 500 
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the data
text = open("ramayana_text.txt", "r", encoding="utf-8").read()

# make tokens
itos = {token: ch for token, ch in enumerate(sorted(list(set(text))))}
stoi = {ch: token for token, ch in itos.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long, device=device)
decode = lambda x: "".join([itos[token.item()] for token in x])

# Encode everything once safe computation
data_all = encode(text)
n90 = int(len(text) * 0.90)
train_set = data_all[:n90]
val_set = data_all[n90:]

# Helper function
def get_data(type="train"):
    data = train_set if type == "train" else val_set
    idxs = torch.randint(len(data)-context_size-1, (batch_size, ))
    offsets = torch.arange(context_size)
    x = data[idxs.unsqueeze(dim=1) + offsets]
    y = data[idxs.unsqueeze(dim=1) + offsets + 1]
    
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(m):
    res = {}
    m.eval() # Set the module in evaluation mode    
    for type in ["train", "val"]:
        out = torch.zeros(eval_iter)
        for i in range(eval_iter):
            x, y = get_data(type)
            logits, loss = m(x, y)
            out[i] = loss.item()
        res[type] = out.mean().item()
    m.train() # reset module to train mode
    return res

# instantiate bigram model
m = GPT(vocab_size)
m = m.to(device)

# create Adam optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

# Train bigram model
for i in range(max_iter):

    # get metric about model once in a while
    if i % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Step {i}: Train Loss {losses['train']:.4f} Val Loss {losses['val']:.4f}")

    # forward pass    
    x, y = get_data("train")
    _, loss = m(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # update parameters
    optimizer.step()

# Save the Trained Model for reusability & after generating text 
# (its most neccessory thing in our life we can't train GPT everytime )
torch.save(m.state_dict(), "nano_gpt.pth")

# Store tokenizer also for stay consistent
with open("tokenizer.json", "w") as f:
    json.dump({
        "stoi": stoi,
        "itos": itos
    } , f, ensure_ascii=False)