import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) 
        emb = t[:, None] * emb[None, :]  
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  
        return emb
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, t):
        emb = self.time_emb(t)
        return self.mlp(emb)

def add_gaussian_noise(x0, t, alpha_bar):

    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar[t])

    while sqrt_alpha_bar.dim() < x0.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

    eps = torch.randn_like(x0)

    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * eps

    return x_t, eps

class Attention(nn.Module):
  def __init__(self, in_ch):
    super().__init__()
    self.group_norm = nn.GroupNorm(32, in_ch)

    self.q = nn.Conv2d(in_ch, in_ch, 1)
    self.k = nn.Conv2d(in_ch, in_ch, 1)
    self.v = nn.Conv2d(in_ch, in_ch, 1)

    self.proj = nn.Conv2d(in_ch, in_ch, 1)
  def forward(self, x) -> torch.Tensor:
    b, c, h, w = x.shape
    x1 = x
    x = self.group_norm(x)

    q = self.q(x).view(b, c, h*w).permute(0, 2, 1)
    k = self.k(x).view(b, c, h*w)
    v = self.v(x).view(b, c, h*w).permute(0, 2, 1)
    attn = torch.matmul(q, k) * (c ** -0.5)
    A = F.softmax(attn, dim = -1)
    out = torch.matmul(A, v)
    out = out.permute(0, 2, 1).reshape(b, c, h, w)

    scores = self.proj(out)

    return x1 + scores



class DownSample(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1)
  def forward(self, x) -> torch.Tensor:
    return self.conv1(x)
class UpSample(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode = 'nearest'),
        nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding=1)
    )
  def forward(self, x) -> torch.Tensor:
    return self.up(x)


class ResBlock(nn.Module):
  def __init__(self, in_ch, out_ch, embed_dim, p = 0.2):
    super().__init__()
    self.norm1 = nn.GroupNorm(32, in_ch)
    self.act1 = nn.SiLU()
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)

    self.norm2 = nn.GroupNorm(32, out_ch)
    self.act2 = nn.SiLU()
    self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1)
    self.dropout = nn.Dropout(p)
    self.time_proj = nn.Sequential(
        nn.SiLU(),
        nn.Linear(embed_dim, out_ch)
    )

    if in_ch  != out_ch:
      self.skip = nn.Conv2d(in_ch, out_ch, 1)
    else:
      self.skip = nn.Identity()
  def forward(self, x, t_emb) -> torch.Tensor:
    x1 = self.conv1(self.act1(self.norm1(x)))
    x2 = x1 + self.time_proj(t_emb)[:, :, None, None]
    x3 = self.conv2(self.dropout(self.act2(self.norm2(x2))))
    return x3 + self.skip(x)


class Bottleneck(nn.Module):
    def __init__(self, ch, embed_dim):
        super().__init__()
        self.res1 = ResBlock(ch, ch, embed_dim)
        self.attn = Attention(ch)
        self.res2 = ResBlock(ch, ch, embed_dim)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, embed_dim=256, time_emb_dim=1024):
        super().__init__()
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.time_mlp = TimeEmbedding(embed_dim)

        self.bottleneck = Bottleneck(base_ch*8, embed_dim)

        self.res1 = ResBlock(base_ch, base_ch, embed_dim)
        self.res2 = ResBlock(base_ch, base_ch, embed_dim)
        self.down1 = DownSample(base_ch, base_ch*2)

        self.res3 = ResBlock(base_ch*2, base_ch*2, embed_dim)
        self.res4 = ResBlock(base_ch*2, base_ch*2, embed_dim)
        self.down2 = DownSample(base_ch*2, base_ch*4)

        self.res5 = ResBlock(base_ch*4, base_ch*4, embed_dim)
        self.res6 = ResBlock(base_ch*4, base_ch*4, embed_dim)
        self.down3 = DownSample(base_ch*4, base_ch*8)


        self.up1 = UpSample(base_ch*8, base_ch*4)
        self.res9 = ResBlock(base_ch*8, base_ch*4, embed_dim)
        self.res10 = ResBlock(base_ch*4, base_ch*4, embed_dim)

        self.up2 = UpSample(base_ch*4, base_ch*2)
        self.res11 = ResBlock(base_ch*4, base_ch*2, embed_dim)
        self.res12 = ResBlock(base_ch*2, base_ch*2, embed_dim)

        self.up3 = UpSample(base_ch*2, base_ch)
        self.res13 = ResBlock(base_ch*2, base_ch, embed_dim)
        self.res14 = ResBlock(base_ch, base_ch, embed_dim)
        
        self.attn = Attention(base_ch*2)

        self.final_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        x1 = self.res1(x, t_emb)
        x1 = self.res2(x1, t_emb)
        x_down1 = self.down1(x1)

        x2 = self.res3(x_down1, t_emb)
        x2 = self.res4(x2, t_emb)
        x2 = self.attn(x2)
        x_down2 = self.down2(x2)

        x3 = self.res5(x_down2, t_emb)
        x3 = self.res6(x3, t_emb)
        x_down3 = self.down3(x3)

        x4 = self.bottleneck(x_down3, t_emb)

        x_up1 = self.up1(x4)
        x_up1 = torch.cat([x_up1, x3], dim=1)
        x_up1 = self.res9(x_up1, t_emb)
        x_up1 = self.res10(x_up1, t_emb)

        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.res11(x_up2, t_emb)
        x_up2 = self.res12(x_up2, t_emb)

        x_up3 = self.up3(x_up2)
        x_up3 = torch.cat([x_up3, x1], dim=1)
        x_up3 = self.res13(x_up3, t_emb)
        x_up3 = self.res14(x_up3, t_emb)

        return self.final_conv(x_up3)
model = UNet()
device = torch.device('cuda')
train = False
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
num_epochs = 200
T = 1000
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


dataset = datasets.ImageFolder(
    root="/kaggle/input/cats-faces-64x64-for-generative-models",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
ema_decay = 0.999
ema_model = copy.deepcopy(model)
for p in ema_model.parameters():
    p.requires_grad_(False) 
ema_model = ema_model.to(device)
betas = torch.linspace(1e-4, 0.02, T, device=device, dtype = torch.float32)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
model = model.to(device)
model.train()
if train:
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for x0, _ in loop:
            x0 = x0.to(device).float()
            x0 = x0 * 2 - 1
    
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = add_gaussian_noise(x0, t, alpha_bar)
            eps_pred = model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1 - ema_decay))
    
            loop.set_postfix(loss=loss.item())
            torch.save(ema_model.state_dict(), f"/kaggle/working/ema_model{epoch}.pth")

        print(f"epoch {epoch} | loss {loss.item():.4f}")


ema_model.load_state_dict(torch.load("/kaggle/working/ema_model12.pth", map_location=device))
ema_model.eval()

@torch.no_grad()
def p_sample_ddim(model, x, t, t_prev, alpha_bar):
    eps = model(x, t)

    alpha_bar_t = alpha_bar[t][:, None, None, None]
    alpha_bar_prev = alpha_bar[t_prev][:, None, None, None]

    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

    x_prev = (
        torch.sqrt(alpha_bar_prev) * x0_pred +
        torch.sqrt(1 - alpha_bar_prev) * eps
    )

    return x_prev

@torch.no_grad()
def sample_ddim(
    model,
    image_size=64,
    batch_size=16,
    channels=3,
    device="cuda",
    ddim_steps=50
):
    model.eval()

    times = torch.linspace(
        T - 1, 0, ddim_steps,
        device=device,
        dtype=torch.long
    )

    x = torch.randn(
        batch_size,
        channels,
        image_size,
        image_size,
        device=device
    )

    for i in range(len(times) - 1):
        t = times[i].repeat(batch_size)
        t_prev = times[i + 1].repeat(batch_size)

        x = p_sample_ddim(model, x, t, t_prev, alpha_bar)

    return x

samples = sample_ddim(
    ema_model,
    image_size=64,
    batch_size=8,
    device=device
)
import matplotlib.pyplot as plt

grid = samples.cpu().permute(0, 2, 3, 1)

plt.figure(figsize=(8, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    img = (grid[i] + 1) / 2
    img = img.clamp(0, 1)
    plt.imshow(img)
    plt.axis("off")
plt.show()