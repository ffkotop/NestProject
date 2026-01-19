import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import triton
import triton.language as tl
import time
from torch.amp import GradScaler, autocast
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

embed_dim = 256
hidden_dim = 1024
num_heads = 8
num_layers = 8
seq_len = 128
dropout = 0.15
bias = False



def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "silu":
        accumulator = accumulator * tl.sigmoid(accumulator)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

class TorchFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        x = self.w1(x)
        x = F.silu(x)
        x = self.w2(x)
        return x

class TritonFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((embed_dim, hidden_dim), dtype=torch.float16, device=DEVICE))
        self.w2 = nn.Parameter(torch.empty((hidden_dim, embed_dim), dtype=torch.float16, device=DEVICE))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        return matmul(matmul(x, self.w1, activation="silu"), self.w2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, use_triton = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_triton = use_triton
        if use_triton:
            self.ffn = TritonFFN(embed_dim, hidden_dim)
        else: 
            self.ffn = TorchFFN(embed_dim, hidden_dim)

    def forward(self, x, attn_mask, key_padding_mask):
        x = x.to(DEVICE)
        dtype = x.dtype
        B, S, K = x.shape

        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + attn_out
        if self.use_triton:
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm.reshape(-1, K).half()).reshape(B, S, K)
            x = x + ffn_out
        else:
            x = self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.embedding(idx)
        x = self.pos_enc(x)
        x = self.dropout(x)

        key_padding_mask = (idx == tokenizer.pad_token_id)
        seq_len = T
        causal_mask = self.causal_mask[:T, :T]

        for block in self.blocks:
            x = block(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask
            )

        x = self.norm(x)
        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        return logits, loss
model = Transformer().to(device)

print(f"Параметры: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
num_epochs = 4

scaler = GradScaler('cuda')

class LMTextDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, "r", encoding="utf-16") as f:
            text = f.read()

        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        self.seq_len = seq_len
        self.samples = []

        for i in range(0, len(tokens) - seq_len, seq_len):
            x = tokens[i:i+seq_len]
            y = tokens[i+1:i+seq_len+1]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


#data_train = []
#with open("drive/MyDrive/subsetsmall.txt", mode = "r") as file:
#  for i, line in enumerate(file):
#    if i >= 120000:
#      break
#    data_train.append(line)
#with open("drive/MyDrive/subsetsmall.txt", mode = "w", encoding = "utf-8") as file:
#  file.writelines(line for line in data_train)


dataset = LMTextDataset("/kaggle/input/5kkenglsen/clean1.txt", tokenizer, seq_len)

batch_size = 32
accumulation_steps = 1
effective_batch_size = batch_size * accumulation_steps



dataloader = DataLoader(
    dataset,
    batch_size=batch_size, 
    pin_memory=True,
    num_workers=4
)


train = True

model.train()
if train:
    for epoch in range(num_epochs):     
        total_loss = 0
        optimizer.zero_grad()
    
        for i, (x, y) in enumerate(tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}" 
        )):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
    
            with autocast('cuda'):
                _, loss = model(x, y)
                loss = loss / accumulation_steps
    
            scaler.scale(loss).backward()
    
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
            total_loss += loss.item() * accumulation_steps
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"/kaggle/working/model{epoch}.pth")


print("Готово! Модель сохранена.")

state_dict = torch.load("/kaggle/working/model1", map_location="cpu")

model.eval()


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    stop_token: str = None
):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        current_seq = generated[:, -seq_len:]

        with autocast('cuda'):
            logits = model(current_seq)
        next_token_logits = logits[:, -1, :]
        if repetition_penalty != 1.0:
            for i in range(generated.size(0)):
                for token in generated[i].tolist():
                    if next_token_logits[i, token] < 0:
                        next_token_logits[i, token] *= repetition_penalty
                    else:
                        next_token_logits[i, token] /= repetition_penalty

        next_token_logits = next_token_logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.where(
                next_token_logits < top_k_logits[:, -1:],
                torch.tensor(float('-inf'), device=device),
                next_token_logits
            )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if stop_token and tokenizer.decode(next_token[0], skip_special_tokens=False) == stop_token:
            break
        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output
print(generate("Now we are"))