"""Native PyTorch ModernBERT and Laya decision heads (inference only).

Architecture follows Laya and Hugging Face ModernBERT; see NOTICE. No MLX
operations are used by this implementation.
"""

from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    @classmethod
    def from_dict(cls, value: dict):
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.model_type != "modernbert":
            raise ValueError(f"Unsupported encoder: {cfg.model_type!r}; expected modernbert")
        if cfg.hidden_activation != "gelu":
            raise ValueError(f"Unsupported encoder activation: {cfg.hidden_activation!r}")
        if cfg.hidden_size % cfg.num_attention_heads or cfg.head_dim % 2:
            raise ValueError("ModernBERT requires an even, integral attention head dimension")
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid ModernBERT layer_types")
        for kind in set(cfg.layer_types):
            params = (cfg.rope_parameters or {}).get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ValueError("Only default (unscaled) ModernBERT RoPE is supported")
        return cfg

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind):
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


def apply_rotary_pos_emb(x, base):
    b, num_heads, length, head_dim = x.shape
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    t = torch.arange(length, device=x.device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq) # [length, half]
    emb = torch.cat((freqs, freqs), dim=-1) # [length, head_dim]
    cos = emb.cos().view(1, 1, length, head_dim)
    sin = emb.sin().view(1, 1, length, head_dim)
    
    # MLX traditional=False rotates elements [i] with [i + half]
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    # Cast cos/sin to match x dtype
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + x_rotated * sin


class EncoderAttention(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.base = cfg.rope_base(kind)
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x, mask):
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        # qkv is [b, length, 3, num_heads, head_dim]
        # we want [b, num_heads, length, head_dim] for F.scaled_dot_product_attention
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        
        q = apply_rotary_pos_emb(q, self.base)
        k = apply_rotary_pos_emb(k, self.base)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # out is [b, num_heads, length, head_dim]
        out = out.transpose(1, 2).reshape(b, length, -1)
        return self.Wo(out)


class EncoderMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


def attention_masks(attention_mask, window):
    """Boolean key masks, with inclusive local distance <= local_attention // 2."""
    valid = attention_mask.bool()
    b, length = valid.shape
    # PyTorch SDPA expects boolean mask where True means participate in attention
    full = valid.view(b, 1, 1, length) # [b, 1, 1, length]
    
    positions = torch.arange(length, device=valid.device)
    dist = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
    local = dist <= window // 2 # [length, length]
    
    local = local.unsqueeze(0).unsqueeze(0) # [1, 1, length, length]
    # Allow padded queries to see valid keys so they don't get NaN softmax
    # MLX logic: local = (local | ~valid_queries) & valid_keys
    # In PyTorch, padded queries can just see all valid keys (or just themselves). Let's match MLX exactly:
    valid_q = valid.view(b, 1, length, 1)
    local = (local | ~valid_q) & full
    
    return {"full_attention": full, "sliding_attention": local}


class ModernBert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList([EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.num_heads = max(1, dims // 64)
        if dims % self.num_heads:
            raise ValueError("Decision head dimensions must be divisible by its head count")
        self.head_dim = dims // self.num_heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x, mask):
        b, length, _ = x.shape
        qkv = self.in_proj(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, length, -1)
        return self.out_proj(out)


class HeadLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.self_attn = HeadAttention(dims)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecisionHead(nn.Module):
    def __init__(self, dims, count):
        super().__init__()
        self.layers = nn.ModuleList([HeadLayer(dims) for _ in range(count)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecisionModel(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, agent_config: dict):
        super().__init__()
        dims = encoder_config.hidden_size
        self.encoder = ModernBert(encoder_config)
        self.head = DecisionHead(dims, agent_config.get("head_layers", 2))
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(dims + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent_config.get("act_costs", {})) + 1),
        )

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype).unsqueeze(1)
        
        # Head mask shape [b, 1, 1, length]
        head_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        h = self.head(h, head_mask)
        
        b = h.shape[0]
        batch_idx = torch.arange(b, device=h.device).unsqueeze(1)
        pos = torch.clamp(marker_pos, min=0).long()
        markers = h[batch_idx, pos]
        
        logits = self.scorer(markers).squeeze(-1)
        # Ensure it is float32 for softmax stability
        logits = logits.float()
        logits = torch.where(marker_mask.bool(), logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))
        
        p = torch.softmax(logits, dim=-1)
        k = torch.clamp(marker_mask.sum(dim=-1), min=2).float()
        
        entropy = -(p * torch.log(torch.clamp(p, min=1e-9))).sum(dim=-1) / torch.log(k)
        
        top = torch.sort(p, dim=-1)[0][:, -2:]
        features = torch.stack([top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0], dim=-1)
        
        pooled = torch.cat([h[:, 0].float(), features], dim=-1)
        action = self.act_head(pooled.to(self.act_head[0].weight.dtype))
        return logits, action.float()


def sanitize_weights(weights):
    """Return PyTorch weights as is. The original upstream was PyTorch."""
    return weights
