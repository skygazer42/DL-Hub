import math
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.utils import RMSNorm, _d, build_sinusoidal_positions, masked_mean_pool


def _expand_key_padding_mask(mask: torch.Tensor, *, b: int, t: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"attention_mask must be (B, T), got {tuple(mask.shape)}")
    if mask.shape != (b, t):
        raise ValueError(
            f"attention_mask shape mismatch: expected {(b, t)}, got {tuple(mask.shape)}"
        )
    return mask.to(torch.bool).view(b, 1, 1, t)


def _make_causal_mask(t: int, *, device: torch.device) -> torch.Tensor:
    # (1, 1, T, T) bool mask; True means allowed.
    m = torch.ones((t, t), device=device, dtype=torch.bool).tril()
    return m.view(1, 1, t, t)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, base: float = 10000.0) -> None:
        super().__init__()
        d = int(head_dim)
        if d % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B, H, T, Hd)
        b, h, t, d = q.shape
        if k.shape != (b, h, t, d):
            raise ValueError("q and k shapes must match for RoPE")
        pos = torch.arange(t, device=q.device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", pos, self.inv_freq)  # (T, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d)
        cos = emb.cos().view(1, 1, t, d)
        sin = emb.sin().view(1, 1, t, d)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    # From the ALiBi paper reference implementation (simplified).
    n = int(num_heads)
    if n <= 0:
        raise ValueError("num_heads must be > 0")

    def _get_power_of_2_slopes(m: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** (-(math.log2(m) - 3))))
        ratio = start
        return [start * (ratio**i) for i in range(m)]

    if math.log2(n).is_integer():
        slopes = _get_power_of_2_slopes(n)
    else:
        closest = 2 ** int(math.floor(math.log2(n)))
        slopes = _get_power_of_2_slopes(closest)
        slopes_extra = _get_power_of_2_slopes(2 * closest)
        slopes.extend(slopes_extra[0::2][: n - closest])
    return torch.tensor(slopes, dtype=torch.float32)  # (H,)


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, *, max_distance: int) -> None:
        super().__init__()
        h = int(num_heads)
        md = int(max_distance)
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        if md <= 0:
            raise ValueError("max_distance must be > 0")
        self.num_heads = h
        self.max_distance = md
        self.bias = nn.Embedding(2 * md + 1, h)

    def forward(self, t: int, *, device: torch.device) -> torch.Tensor:
        # returns (1, H, T, T)
        pos = torch.arange(t, device=device)
        rel = pos[None, :] - pos[:, None]  # (T, T)
        rel = rel.clamp(min=-self.max_distance, max=self.max_distance) + self.max_distance
        b = self.bias(rel)  # (T, T, H)
        return b.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)


class FFN(nn.Module):
    def __init__(self, dim: int, *, mult: int, dropout: float, kind: str) -> None:
        super().__init__()
        d = int(dim)
        hidden = max(8, int(d * int(mult)))
        kind = str(kind).lower().strip()

        if kind in {"relu"}:
            self.net = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, d),
                nn.Dropout(p=float(dropout)),
            )
        elif kind in {"gelu"}:
            self.net = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, d),
                nn.Dropout(p=float(dropout)),
            )
        elif kind in {"swiglu", "geglu"}:
            act = "silu" if kind == "swiglu" else "gelu"
            self.fc = nn.Linear(d, 2 * hidden)
            self.proj = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, d),
                nn.Dropout(p=float(dropout)),
            )
            self.act = act
        else:
            raise ValueError("Unknown FFN kind. Supported: relu|gelu|swiglu|geglu")

        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in {"relu", "gelu"}:
            return self.net(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        if self.act == "silu":
            y = torch.nn.functional.silu(a) * b
        else:
            y = torch.nn.functional.gelu(a) * b
        return self.proj(y)


class QKVScores(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_kv_heads: int | None,
        dropout: float,
        use_rope: bool,
        use_alibi: bool,
        rel_bias: RelativePositionBias | None,
        causal: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)

        kvh = h if num_kv_heads is None else int(num_kv_heads)
        if kvh <= 0 or h % kvh != 0:
            kvh = h
        self.num_kv_heads = kvh
        self.kv_repeat = int(h // kvh)

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kvh * self.head_dim, bias=False)
        self.v = nn.Linear(d, kvh * self.head_dim, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))

        self.use_rope = bool(use_rope)
        self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None

        self.use_alibi = bool(use_alibi)
        if self.use_alibi:
            slopes = _alibi_slopes(h)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.register_buffer("alibi_slopes", torch.empty(0), persistent=False)

        self.rel_bias = rel_bias
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = self.k(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B,kvH,T,Hd)
        v = self.v(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_repeat != 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        if self.use_rope:
            if self.rope is None:
                raise RuntimeError("use_rope=True but rope module missing")
            q, k = self.rope(q, k)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)

        # Add biases.
        if self.use_alibi:
            pos = torch.arange(t, device=x.device, dtype=torch.float32)
            rel = pos[None, :] - pos[:, None]  # (T,T)
            bias = -rel.abs().unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            scores = scores + slopes * bias

        if self.rel_bias is not None:
            scores = scores + self.rel_bias(t, device=x.device)

        key_mask = _expand_key_padding_mask(attention_mask, b=b, t=t)
        scores = scores.masked_fill(~key_mask, -1e9)

        if self.causal:
            causal = _make_causal_mask(t, device=x.device)
            scores = scores.masked_fill(~causal, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class LinformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        proj_k: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)
        self.proj_k = int(proj_k)
        if self.proj_k <= 0:
            raise ValueError("proj_k must be > 0")

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))
        self.causal = bool(causal)

        # Learnable anchor locations in [0, 1] (after sigmoid), used to softly
        # pool the full sequence into K "landmark" tokens.
        self.anchors = nn.Parameter(torch.linspace(-2.0, 2.0, steps=self.proj_k))

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, Hd) -> (B, H, K, Hd)
        b, h, t, d = x.shape
        pos = torch.linspace(0.0, 1.0, steps=t, device=x.device, dtype=x.dtype).view(1, 1, 1, t)
        anchors = (
            torch.sigmoid(self.anchors)
            .to(device=x.device, dtype=x.dtype)
            .view(1, 1, self.proj_k, 1)
        )  # (1,1,K,1)
        dist2 = (pos - anchors).pow(2)  # (1,1,K,T)

        # Soft selection (stable for small T): weights over T for each of K anchors.
        weights = torch.softmax(-dist2 * 16.0, dim=-1)  # (1,1,K,T)
        weights = weights.expand(b, h, -1, -1)
        return torch.einsum("bhtd,bhkt->bhkd", x, weights)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        key_mask = attention_mask.to(torch.bool).view(b, 1, t, 1)
        k = k * key_mask.to(k.dtype)
        v = v * key_mask.to(v.dtype)

        k = self._project(k)
        v = self._project(v)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,K)
        if self.causal:
            # Linformer causal masking (approx): prevent attending to projected keys from future.
            scores = scores  # best-effort; keep simple for fixed small T
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class PerformerSelfAttention(nn.Module):
    """Kernelized linear attention (Performer-style), simplified with ELU+1 kernel."""

    def __init__(self, dim: int, *, num_heads: int, dropout: float, causal: bool) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))
        self.causal = bool(causal)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        mask = attention_mask.to(torch.float32).view(b, 1, t, 1)
        k = k * mask
        v = v * mask

        qf = self._phi(q)
        kf = self._phi(k)

        if self.causal:
            # Causal linear attention via prefix sums.
            kv = torch.cumsum(kf.unsqueeze(-1) * v.unsqueeze(-2), dim=2)  # (B,H,T,Hd,Hd)
            k1 = torch.cumsum(kf, dim=2)  # (B,H,T,Hd)
            out = torch.einsum("bhtd,bhtde->bhte", qf, kv)
            denom = torch.einsum("bhtd,bhtd->bht", qf, k1).unsqueeze(-1).clamp(min=1e-6)
            out = out / denom
        else:
            kv = torch.einsum("bhtd,bhte->bhde", kf, v)  # (B,H,Hd,Hd)
            k1 = kf.sum(dim=2)  # (B,H,Hd)
            out = torch.einsum("bhtd,bhde->bhte", qf, kv)
            denom = torch.einsum("bhtd,bhd->bht", qf, k1).unsqueeze(-1).clamp(min=1e-6)
            out = out / denom

        out = self.drop(out)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class LongformerSelfAttention(nn.Module):
    """Sliding window attention (Longformer-style), simplified."""

    def __init__(
        self, dim: int, *, num_heads: int, window: int, dropout: float, causal: bool
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)
        self.window = int(window)
        if self.window <= 0:
            raise ValueError("window must be > 0")

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        key_mask = attention_mask.to(torch.bool)  # (B,T)
        out = torch.zeros(
            (b, self.num_heads, t, self.head_dim), device=x.device, dtype=torch.float32
        )
        scale = float(self.head_dim) ** -0.5
        w = int(self.window)

        for i in range(t):
            j0 = max(0, i - w)
            j1 = min(t, i + w + 1)
            if self.causal:
                j1 = min(j1, i + 1)
            ki = k[:, :, j0:j1, :]  # (B,H,win,Hd)
            vi = v[:, :, j0:j1, :]
            qi = q[:, :, i : i + 1, :]  # (B,H,1,Hd)
            scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale  # (B,H,1,win)
            km = key_mask[:, j0:j1].view(b, 1, 1, -1)
            scores = scores.masked_fill(~km, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out[:, :, i : i + 1, :] = torch.matmul(attn, vi)

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class TalkingHeadsSelfAttention(nn.Module):
    """Talking-Heads attention (Shazeer et al.), simplified."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float,
        use_rope: bool,
        use_alibi: bool,
        rel_bias: RelativePositionBias | None,
        causal: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))

        self.use_rope = bool(use_rope)
        self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None

        self.use_alibi = bool(use_alibi)
        if self.use_alibi:
            slopes = _alibi_slopes(h)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.register_buffer("alibi_slopes", torch.empty(0), persistent=False)

        self.rel_bias = rel_bias
        self.causal = bool(causal)

        self.proj_logits = nn.Linear(h, h, bias=False)
        self.proj_attn = nn.Linear(h, h, bias=False)

        nn.init.eye_(self.proj_logits.weight)
        nn.init.eye_(self.proj_attn.weight)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if self.rope is None:
                raise RuntimeError("use_rope=True but rope module missing")
            q, k = self.rope(q, k)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)

        if self.use_alibi:
            pos = torch.arange(t, device=x.device, dtype=torch.float32)
            rel = pos[None, :] - pos[:, None]
            bias = -rel.abs().unsqueeze(0).unsqueeze(0)
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            scores = scores + slopes * bias

        if self.rel_bias is not None:
            scores = scores + self.rel_bias(t, device=x.device)

        # Talking-heads projection on logits across heads.
        scores = self.proj_logits(scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        key_mask = _expand_key_padding_mask(attention_mask, b=b, t=t)
        scores = scores.masked_fill(~key_mask, -1e9)
        if self.causal:
            causal = _make_causal_mask(t, device=x.device)
            scores = scores.masked_fill(~causal, -1e9)

        attn = torch.softmax(scores, dim=-1)
        # Talking-heads projection on attention weights across heads.
        attn = self.proj_attn(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.masked_fill(~key_mask, 0.0)
        if self.causal:
            attn = attn.masked_fill(~causal, 0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class SynthesizerSelfAttention(nn.Module):
    """Synthesizer attention (Tay et al.), simplified."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        seq_len: int,
        dropout: float,
        mode: str,
        hidden_dim: int,
        causal: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)
        self.seq_len = int(seq_len)
        self.causal = bool(causal)

        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))

        mode = str(mode).lower().strip()
        if mode not in {"random", "mlp"}:
            raise ValueError("mode must be random|mlp")
        self.mode = mode

        if self.mode == "random":
            self.w = nn.Parameter(torch.randn(h, self.seq_len, self.seq_len) * 0.02)
            self.mlp = None
        else:
            hidden = int(hidden_dim)
            if hidden <= 0:
                raise ValueError("hidden_dim must be > 0")
            self.w = None
            self.mlp = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, h * self.seq_len),
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        if t != self.seq_len:
            raise ValueError(f"Synthesizer expected seq_len={self.seq_len}, got T={t}")

        v = self.v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        if self.mode == "random":
            if self.w is None:
                raise RuntimeError("random synthesizer requested but w is None")
            scores = self.w.unsqueeze(0).expand(b, -1, -1, -1)  # (B,H,T,T)
        else:
            if self.mlp is None:
                raise RuntimeError("mlp synthesizer requested but mlp is None")
            scores = self.mlp(x).view(b, t, self.num_heads, t).permute(0, 2, 1, 3).contiguous()

        key_mask = _expand_key_padding_mask(attention_mask, b=b, t=t)
        scores = scores.masked_fill(~key_mask, -1e9)
        if self.causal:
            causal = _make_causal_mask(t, device=x.device)
            scores = scores.masked_fill(~causal, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class NystromSelfAttention(nn.Module):
    """Nyström approximation of softmax attention (Nyströmformer), simplified."""

    def __init__(
        self, dim: int, *, num_heads: int, num_landmarks: int, dropout: float, causal: bool
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)
        self.num_landmarks = int(num_landmarks)
        if self.num_landmarks <= 0:
            raise ValueError("num_landmarks must be > 0")
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))
        self.causal = bool(causal)

    def _segment_mean(
        self, x: torch.Tensor, mask: torch.Tensor, *, num_segments: int
    ) -> torch.Tensor:
        # x: (B,H,T,Hd), mask: (B,T)
        b, h, t, d = x.shape
        m = int(num_segments)
        seg = int(math.ceil(t / float(m)))
        out = torch.zeros((b, h, m, d), device=x.device, dtype=torch.float32)
        w = mask.to(torch.float32).view(b, 1, t, 1)
        for i in range(m):
            j0 = i * seg
            j1 = min(t, (i + 1) * seg)
            if j0 >= t:
                continue
            ww = w[:, :, j0:j1, :]
            denom = ww.sum(dim=2).clamp(min=1.0)
            out[:, :, i, :] = (x[:, :, j0:j1, :] * ww).sum(dim=2) / denom
        return out

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        q_land = self._segment_mean(
            q, attention_mask, num_segments=self.num_landmarks
        )  # (B,H,M,Hd)
        k_land = self._segment_mean(k, attention_mask, num_segments=self.num_landmarks)

        # A: (B,H,T,M)
        a_scores = torch.matmul(q, k_land.transpose(-2, -1)) * scale
        a = torch.softmax(a_scores, dim=-1)

        # B: (B,H,M,M)
        b_scores = torch.matmul(q_land, k_land.transpose(-2, -1)) * scale
        b_attn = torch.softmax(b_scores, dim=-1)
        b_inv = torch.linalg.pinv(b_attn.to(torch.float32))

        # C: (B,H,M,T) (mask keys)
        c_scores = torch.matmul(q_land, k.transpose(-2, -1)) * scale
        key_mask = _expand_key_padding_mask(attention_mask, b=b, t=t)
        c_scores = c_scores.masked_fill(
            ~key_mask.view(b, 1, 1, t).expand(b, self.num_heads, -1, -1), -1e9
        )
        if self.causal:
            causal = _make_causal_mask(t, device=x.device)
            c_scores = c_scores.masked_fill(
                ~causal.view(1, 1, t, t)[:, :, : self.num_landmarks], -1e9
            )
        c = torch.softmax(c_scores, dim=-1)

        cv = torch.matmul(c, v)  # (B,H,M,Hd)
        ab = torch.matmul(a, b_inv)  # (B,H,T,M)
        out = torch.matmul(ab, cv)  # (B,H,T,Hd)
        out = self.drop(out)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class BigBirdSelfAttention(nn.Module):
    """Block-sparse attention (BigBird-style), simplified for small fixed T."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        seq_len: int,
        window: int,
        num_random: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = int(d // h)
        self.seq_len = int(seq_len)
        self.window = int(window)
        self.num_random = int(num_random)
        if self.window < 0 or self.num_random < 0:
            raise ValueError("window/num_random must be >= 0")

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(p=float(dropout))
        self.causal = bool(causal)

        # Precompute per-token sparse indices (fixed length).
        max_len = 1 + (2 * self.window + 1) + self.num_random
        idx = torch.zeros((self.seq_len, max_len), dtype=torch.long)
        idx_mask = torch.zeros((self.seq_len, max_len), dtype=torch.bool)
        g = torch.Generator()
        g.manual_seed(0)
        for i in range(self.seq_len):
            ids: list[int] = []
            # global token 0
            ids.append(0)
            # local window
            for j in range(i - self.window, i + self.window + 1):
                if 0 <= j < self.seq_len:
                    ids.append(j)
            # random tokens
            if self.num_random > 0:
                perm = torch.randperm(self.seq_len, generator=g).tolist()
                for j in perm:
                    if j not in ids:
                        ids.append(int(j))
                    if len(ids) >= max_len:
                        break
            ids = ids[:max_len]
            idx[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            idx_mask[i, : len(ids)] = True

        self.register_buffer("_idx", idx, persistent=False)
        self.register_buffer("_idx_mask", idx_mask, persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        if t != self.seq_len:
            raise ValueError(f"BigBird expected seq_len={self.seq_len}, got T={t}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        key_mask_full = attention_mask.to(torch.bool)  # (B,T)
        out = torch.zeros(
            (b, self.num_heads, t, self.head_dim), device=x.device, dtype=torch.float32
        )
        scale = float(self.head_dim) ** -0.5

        for i in range(t):
            idx = self._idx[i].to(device=x.device)
            valid = self._idx_mask[i].to(device=x.device)  # (L,)

            if self.causal:
                valid = valid & (idx <= i)

            ki = k[:, :, idx, :]  # (B,H,L,Hd)
            vi = v[:, :, idx, :]
            qi = q[:, :, i : i + 1, :]  # (B,H,1,Hd)
            scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale  # (B,H,1,L)

            km = key_mask_full[:, idx].view(b, 1, 1, -1)
            vm = valid.view(1, 1, 1, -1)
            scores = scores.masked_fill(~(km & vm), -1e9)

            attn = torch.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out[:, :, i : i + 1, :] = torch.matmul(attn, vi)

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


def _make_norm(kind: str, dim: int) -> nn.Module:
    k = str(kind).lower().strip()
    if k in {"layer", "layernorm", "ln"}:
        return nn.LayerNorm(int(dim))
    if k in {"rms", "rmsnorm"}:
        return RMSNorm(int(dim))
    raise ValueError("Unknown norm kind. Supported: layer|rms")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float,
        attn: nn.Module,
        ffn_kind: str,
        norm_kind: str,
        prenorm: bool,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.prenorm = bool(prenorm)
        self.norm1 = _make_norm(norm_kind, d)
        self.norm2 = _make_norm(norm_kind, d)
        self.attn = attn
        self.drop1 = nn.Dropout(p=float(dropout))
        self.ff = FFN(d, mult=4, dropout=float(dropout), kind=str(ffn_kind))
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            y = self.norm1(x)
            x = x + self.drop1(self.attn(y, attention_mask))
            z = self.norm2(x)
            x = x + self.drop2(self.ff(z))
            return x

        # post-norm
        y = self.attn(x, attention_mask)
        x = self.norm1(x + self.drop1(y))
        z = self.ff(x)
        x = self.norm2(x + self.drop2(z))
        return x


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_heads: int
    num_layers: int
    pos: str  # learned|sin|none
    rope: bool
    alibi: bool
    rel_bias: bool
    attn_impl: str  # full|linformer|performer|longformer|talking_heads|synthesizer|nystrom|bigbird
    num_kv_heads: int | None
    linformer_k: int
    longformer_window: int
    ffn_kind: str
    norm_kind: str
    prenorm: bool
    causal: bool
    pool: str  # mean|cls|attn
    share_layers: bool = False
    # Extra sparse/approx attention knobs (only used when selected by attn_impl).
    synthesizer_mode: str = "random"  # random|mlp
    synthesizer_hidden: int = 128
    nystrom_landmarks: int = 8
    bigbird_window: int = 4
    bigbird_num_random: int = 4


class TransformerTextClassifier(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=64, divisor=8)
        h = int(cfg.num_heads)
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        if d % h != 0:
            h = 1
        self.dim = int(d)
        self.num_heads = int(h)

        self.pool = str(cfg.pool).lower().strip()
        if self.pool not in {"mean", "cls", "attn"}:
            raise ValueError("pool must be one of: mean|cls|attn")

        self.seq_len = int(cfg.max_length) + (1 if self.pool == "cls" else 0)

        self.token = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        pos_kind = str(cfg.pos).lower().strip()
        if pos_kind == "learned":
            self.pos = nn.Embedding(int(self.seq_len), int(d))
            self.register_buffer("_sin_pos", torch.empty(0), persistent=False)
        elif pos_kind == "sin":
            self.pos = None
            self.register_buffer(
                "_sin_pos",
                build_sinusoidal_positions(
                    self.seq_len, int(d), device=torch.device("cpu"), dtype=torch.float32
                ),
                persistent=False,
            )
        elif pos_kind == "none":
            self.pos = None
            self.register_buffer("_sin_pos", torch.empty(0), persistent=False)
        else:
            raise ValueError("pos must be one of: learned|sin|none")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d))) if self.pool == "cls" else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        rel = RelativePositionBias(self.num_heads, max_distance=16) if bool(cfg.rel_bias) else None

        seq_len = int(self.seq_len)
        attn_impl = str(cfg.attn_impl).lower().strip()
        if attn_impl == "full":

            def attn_factory() -> nn.Module:
                return QKVScores(
                    int(d),
                    num_heads=int(self.num_heads),
                    num_kv_heads=cfg.num_kv_heads,
                    dropout=float(cfg.dropout),
                    use_rope=bool(cfg.rope),
                    use_alibi=bool(cfg.alibi),
                    rel_bias=rel,
                    causal=bool(cfg.causal),
                )

        elif attn_impl == "linformer":

            def attn_factory() -> nn.Module:
                return LinformerSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    proj_k=int(cfg.linformer_k),
                    dropout=float(cfg.dropout),
                    causal=bool(cfg.causal),
                )

        elif attn_impl == "performer":

            def attn_factory() -> nn.Module:
                return PerformerSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    dropout=float(cfg.dropout),
                    causal=bool(cfg.causal),
                )

        elif attn_impl == "longformer":

            def attn_factory() -> nn.Module:
                return LongformerSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    window=int(cfg.longformer_window),
                    dropout=float(cfg.dropout),
                    causal=bool(cfg.causal),
                )

        elif attn_impl in {"talking_heads", "talkingheads"}:

            def attn_factory() -> nn.Module:
                return TalkingHeadsSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    dropout=float(cfg.dropout),
                    use_rope=bool(cfg.rope),
                    use_alibi=bool(cfg.alibi),
                    rel_bias=rel,
                    causal=bool(cfg.causal),
                )

        elif attn_impl in {"synthesizer", "synth"}:

            def attn_factory() -> nn.Module:
                return SynthesizerSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    seq_len=seq_len,
                    dropout=float(cfg.dropout),
                    mode=str(cfg.synthesizer_mode),
                    hidden_dim=int(cfg.synthesizer_hidden),
                    causal=bool(cfg.causal),
                )

        elif attn_impl in {"nystrom", "nystromformer"}:

            def attn_factory() -> nn.Module:
                return NystromSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    num_landmarks=int(cfg.nystrom_landmarks),
                    dropout=float(cfg.dropout),
                    causal=bool(cfg.causal),
                )

        elif attn_impl in {"bigbird", "block_sparse"}:

            def attn_factory() -> nn.Module:
                return BigBirdSelfAttention(
                    int(d),
                    num_heads=int(self.num_heads),
                    seq_len=seq_len,
                    window=int(cfg.bigbird_window),
                    num_random=int(cfg.bigbird_num_random),
                    dropout=float(cfg.dropout),
                    causal=bool(cfg.causal),
                )

        else:
            raise ValueError("Unknown attn_impl")

        self.share_layers = bool(cfg.share_layers)
        if self.share_layers:
            self.shared = TransformerBlock(
                int(d),
                num_heads=int(self.num_heads),
                dropout=float(cfg.dropout),
                attn=attn_factory(),
                ffn_kind=str(cfg.ffn_kind),
                norm_kind=str(cfg.norm_kind),
                prenorm=bool(cfg.prenorm),
            )
            self.blocks = nn.ModuleList()
        else:
            self.shared = None
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        int(d),
                        num_heads=int(self.num_heads),
                        dropout=float(cfg.dropout),
                        attn=attn_factory(),
                        ffn_kind=str(cfg.ffn_kind),
                        norm_kind=str(cfg.norm_kind),
                        prenorm=bool(cfg.prenorm),
                    )
                    for _ in range(int(cfg.num_layers))
                ]
            )

        self.norm = _make_norm(str(cfg.norm_kind), int(d))
        self.attn_pool = None
        if self.pool == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(int(d), int(d)),
                nn.Tanh(),
                nn.Linear(int(d), 1, bias=False),
            )
        self.head = nn.Linear(int(d), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        b, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(f"Expected max_length={int(self.cfg.max_length)}, got T={t}")

        x = self.token(input_ids).to(torch.float32)  # (B, T, D)
        if self.pool == "cls":
            if self.cls_token is None:
                raise RuntimeError("cls pooling requested but cls_token is None")
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, D)
            attention_mask = torch.nn.functional.pad(attention_mask, (1, 0), value=1.0)

        if self.pos is not None:
            pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
            x = x + self.pos(pos_ids)
        elif self._sin_pos.numel() != 0:
            pe = self._sin_pos.to(device=x.device, dtype=x.dtype)[: x.shape[1]]
            x = x + pe.unsqueeze(0)
        x = self.drop(x)

        if self.share_layers:
            if self.shared is None:
                raise RuntimeError("share_layers=True but shared block missing")
            for _ in range(int(self.cfg.num_layers)):
                x = self.shared(x, attention_mask)
        else:
            for blk in self.blocks:
                x = blk(x, attention_mask)

        x = self.norm(x)

        if self.pool == "mean":
            pooled = masked_mean_pool(x, attention_mask)
        elif self.pool == "cls":
            pooled = x[:, 0, :]
        else:
            if self.attn_pool is None:
                raise RuntimeError("attn pooling requested but attn_pool missing")
            scores = self.attn_pool(x).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(~attention_mask.to(torch.bool), -1e9)
            w = torch.softmax(scores, dim=1)
            pooled = (w.unsqueeze(-1) * x).sum(dim=1)

        pooled = pooled.to(torch.float32)
        return self.head(pooled)
