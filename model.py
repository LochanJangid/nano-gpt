import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Head + MultiHeadAttention --> Run multiple heads in parallel so the computation will become fast and great use of GPU | i don't have one just using kaggle notebooks:p
class CasualSelfAttention(nn.Module):
    def __init__(self, n_emb, n_head, context_size, dropout):
        super().__init__()
        # RULE: C = head_size * n_head (Everytime don't matter how they run) in our transformer C is n_emb
        self.nh = n_head # number of heads
        self.hs = n_emb//n_head # head size

        self.query = nn.Linear(n_emb, self.nh*self.hs, bias=False) # self.nh*self = n_emb (just for understanding what we are doing...)
        self.key = nn.Linear(n_emb, self.nh*self.hs, bias=False)
        self.value = nn.Linear(n_emb, self.nh*self.hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ! Please give some attention to shapes  hahahahahahahahahaha Very Complicated but that's fun :p
    # x - B, T, C
    # Q - Shapes step by step
    #    B, T, C # first shape
    #    B, T, nh, hs # split C into nh, hs 
    #    B, nh, T, hs # make nh higher dimension because of matrix multiplication (in pytorch matrix multiplication consider last 2 dimensions)
    # k - 
    #    B, T, C -> B, T, nh, hs -> B, nh, T, hs # just same as Q
    #    B, nh, hs, T # transposed shape of k for `Q @ k.T`  [use k.transpose(-1, -2)]
    # v - 
    #    B, T, C -> B, T, nh, hs -> B, nh, T, hs # just same as Q but it is good for (wei @ v) becauzeeeee.... see out shapes :)
    # wei = Q * k.T - (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
    # out = wei @ v - (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    # Reassemble the head back so change shape of out to get back out main shape to stay consistant in shape
    # out - (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh*hs) that is also identical to (B, T, C)
    
    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x) # B, T, C
        Q = Q.view(B, T, self.nh, self.hs) # B, T, nh, hs
        Q = Q.transpose(1, 2) # B, nh, T, hs

        k = self.key(x).view(B, T, self.nh, self.hs) 
        k = k.transpose(1, 2)

        v = self.value(x).view(B, T, self.nh, self.hs)
        v = v.transpose(1, 2)

        wei = Q @ k.transpose(-1, -2) * self.hs ** -0.5 # B, nh, T, T
        # Masking and averaging (if you know you know)
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei) # faaaaaaaaaaaaaaa

        out = wei @ v # B, nh, T, hs
        out = out.transpose(1, 2) # B, T, nh, hs
        out = out.contiguous().view(B, T, self.nh*self.hs) # B, T, nh*hs # used contiguous() becauze after transpose we can't do view directly becauze transpose return non-contiguous tensor 

        out = self.proj(out)        
        out = self.dropout(out)
        return out      

# Feed-Forward just MLP - it gives the space or time to think about collected information
# now tokens think individually on learnt information
class FeedForward(nn.Module):
    def __init__(self, fan_in, fan_out, dropout):
        super().__init__()
        hidden = 4*fan_out
        self.model = nn.Sequential(
            nn.Linear(fan_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fan_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)

# Make a block of transformer where we have communication with computation
# means communication (multi head attention) + computation (feed forward)
class Block(nn.Module):
    def __init__(self, n_emb, n_head, context_size, dropout):
        super().__init__()
        # self.head = MultiHeadAttention(head_size, n_emb//head_size)
        self.head = CasualSelfAttention(n_emb, n_head, context_size, dropout)
        self.ffd = FeedForward(n_emb, n_emb, dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # make residual connection by fork off 
        # here we do layer normalization before attention head & feed forward 
        # it is slighlty different from original transformer architecture but it is better :)
        x = x + self.head(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x

# some configs for our model
@dataclass
class ModelConfig:
    context_size = 256
    n_emb = 384
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_head = 4
    n_layer = 6 
    dropout = 0.2 


# make Biagram model
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.config = ModelConfig()
        cfg = self.config # local alias for config
        self.token_embeddings = nn.Embedding(vocab_size, cfg.n_emb)
        self.position_embeddings = nn.Embedding(cfg.context_size, cfg.n_emb)
        self.blocks = nn.Sequential(*[Block(cfg.n_emb, cfg.n_head, cfg.context_size, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_emb) # final layernorm: a another layer normalization before going outside of transformer block
        self.lm_head = nn.Linear(cfg.n_emb, vocab_size)
        
    def forward(self, idx, target=None):
        # idx - B, T    target - B, T
        B, T = idx.shape
        token_emb = self.token_embeddings(idx) # B, T, C=n_emb
        pos_emb = self.position_embeddings(torch.arange(T, device=self.config.device)) # T, C=n_emb
        x = token_emb + pos_emb # (B, T, C) + (T, C) [broadcast to (B, T, C)] --. B, T, C=n_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, C=n_emb -----Linear Layer----->  B, T, C=vocab_size 

        if target is None:
            loss = None
        else:
            # Cross entropy take B, C but we have B, T, C so we need to stretch out B, T 
            # into B*T sequqnetially so it become B*T, C
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) # 1,

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx - B, T            max_new_tokesn - int
        self.eval()
        for _ in range(max_new_tokens):
            # crop context to max context size
            idx_cond = idx[:, -self.config.context_size:]
            logits, _ = self(idx_cond)
            # logits - B, T, C but we will generate new character on based of last one
            # so we will see last token of logits only
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=1) # B, C
            new_idx = torch.multinomial(probs, num_samples=1) # 1, 
            idx = torch.cat([idx, new_idx], dim=1) # B, T+1
        self.train()
        return idx

