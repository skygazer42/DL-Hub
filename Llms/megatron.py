from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._shared import causal_mask, expand_key_padding_mask, make_attention_mask


def _partition_range(size: int, parallel_size: int, rank: int) -> tuple[int, int]:
    if parallel_size <= 0:
        raise ValueError("tensor_model_parallel_size must be > 0")
    if rank < 0 or rank >= parallel_size:
        raise ValueError("tensor_model_parallel_rank must be within the parallel group")
    if size % parallel_size != 0:
        raise ValueError(f"size {size} must be divisible by tensor_model_parallel_size")
    part = size // parallel_size
    start = rank * part
    return start, start + part


@dataclass(frozen=True)
class MegatronConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    tensor_model_parallel_size: int = 1
    tensor_model_parallel_rank: int = 0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class ColumnParallelLinear(nn.Module):
    is_column_parallel = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        tensor_model_parallel_size: int,
        tensor_model_parallel_rank: int,
        bias: bool = True,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.gather_output = bool(gather_output)
        self.output_start_index, self.output_end_index = _partition_range(
            self.output_size,
            int(tensor_model_parallel_size),
            int(tensor_model_parallel_rank),
        )
        self.output_size_per_partition = self.output_end_index - self.output_start_index
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.bias = (
            nn.Parameter(torch.zeros(self.output_size_per_partition)) if bias else None
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = F.linear(x, self.weight, self.bias)
        if not self.gather_output:
            return local_output
        output = x.new_zeros(*x.shape[:-1], self.output_size)
        output[..., self.output_start_index : self.output_end_index] = local_output
        return output

    def load_full_parameters(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        expected = (self.output_size, self.input_size)
        if tuple(weight.shape) != expected:
            raise ValueError(f"weight must be {expected}, got {tuple(weight.shape)}")
        with torch.no_grad():
            self.weight.copy_(weight[self.output_start_index : self.output_end_index])
            if self.bias is not None:
                if bias is None:
                    raise ValueError("bias is required for a biased ColumnParallelLinear")
                if tuple(bias.shape) != (self.output_size,):
                    raise ValueError(
                        f"bias must be {(self.output_size,)}, got {tuple(bias.shape)}"
                    )
                self.bias.copy_(bias[self.output_start_index : self.output_end_index])

    @staticmethod
    def gather_full_parameters(
        partitions: tuple[ColumnParallelLinear, ...] | list[ColumnParallelLinear],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        parts = tuple(partitions)
        if not parts:
            raise ValueError("partitions must not be empty")
        weight = torch.cat([part.weight for part in parts], dim=0)
        if parts[0].bias is None:
            return weight, None
        bias = torch.cat([part.bias for part in parts if part.bias is not None], dim=0)
        return weight, bias


class RowParallelLinear(nn.Module):
    is_row_parallel = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        tensor_model_parallel_size: int,
        tensor_model_parallel_rank: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.input_is_parallel = bool(input_is_parallel)
        self.input_start_index, self.input_end_index = _partition_range(
            self.input_size,
            int(tensor_model_parallel_size),
            int(tensor_model_parallel_rank),
        )
        self.input_size_per_partition = self.input_end_index - self.input_start_index
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.zeros(self.output_size)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = self.forward_partial(x)
        if self.bias is not None:
            local_output = local_output + self.bias
        return local_output

    def forward_partial(self, x: torch.Tensor) -> torch.Tensor:
        local_input = x
        if not self.input_is_parallel:
            local_input = x[..., self.input_start_index : self.input_end_index]
        return F.linear(local_input, self.weight, None)

    def load_full_parameters(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        expected = (self.output_size, self.input_size)
        if tuple(weight.shape) != expected:
            raise ValueError(f"weight must be {expected}, got {tuple(weight.shape)}")
        with torch.no_grad():
            self.weight.copy_(weight[:, self.input_start_index : self.input_end_index])
            if self.bias is not None:
                if bias is None:
                    raise ValueError("bias is required for a biased RowParallelLinear")
                if tuple(bias.shape) != (self.output_size,):
                    raise ValueError(
                        f"bias must be {(self.output_size,)}, got {tuple(bias.shape)}"
                    )
                self.bias.copy_(bias)

    @staticmethod
    def gather_full_parameters(
        partitions: tuple[RowParallelLinear, ...] | list[RowParallelLinear],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        parts = tuple(partitions)
        if not parts:
            raise ValueError("partitions must not be empty")
        weight = torch.cat([part.weight for part in parts], dim=1)
        bias = None if parts[0].bias is None else parts[0].bias.detach().clone()
        return weight, bias


class VocabParallelEmbedding(nn.Module):
    is_vocab_parallel = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tensor_model_parallel_size: int,
        tensor_model_parallel_rank: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.vocab_start_index, self.vocab_end_index = _partition_range(
            self.num_embeddings,
            int(tensor_model_parallel_size),
            int(tensor_model_parallel_rank),
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(torch.long)
        mask = (ids >= self.vocab_start_index) & (ids < self.vocab_end_index)
        local_ids = (ids - self.vocab_start_index).masked_fill(~mask, 0)
        output = F.embedding(local_ids, self.weight)
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)

    def load_full_weight(self, weight: torch.Tensor) -> None:
        expected = (self.num_embeddings, self.embedding_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(f"weight must be {expected}, got {tuple(weight.shape)}")
        with torch.no_grad():
            self.weight.copy_(weight[self.vocab_start_index : self.vocab_end_index])

    @staticmethod
    def gather_full_weight(
        partitions: tuple[VocabParallelEmbedding, ...] | list[VocabParallelEmbedding],
    ) -> torch.Tensor:
        parts = tuple(partitions)
        if not parts:
            raise ValueError("partitions must not be empty")
        return torch.cat([part.weight for part in parts], dim=0)


class VocabParallelLMHead(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        *,
        vocab_size: int,
        tensor_model_parallel_size: int,
        tensor_model_parallel_rank: int,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.vocab_start_index, self.vocab_end_index = _partition_range(
            self.vocab_size,
            int(tensor_model_parallel_size),
            int(tensor_model_parallel_rank),
        )
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(x, self.weight)
        logits = x.new_zeros(*x.shape[:-1], self.vocab_size)
        logits[..., self.vocab_start_index : self.vocab_end_index] = local_logits
        return logits


class MegatronAttention(nn.Module):
    def __init__(self, config: MegatronConfig) -> None:
        super().__init__()
        if int(config.hidden_size) % int(config.num_attention_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if int(config.num_attention_heads) % int(config.tensor_model_parallel_size) != 0:
            raise ValueError("num_attention_heads must divide tensor_model_parallel_size")

        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.hidden_size_per_attention_head = int(config.hidden_size // config.num_attention_heads)
        self.num_attention_heads_per_partition = int(
            config.num_attention_heads // config.tensor_model_parallel_size
        )
        self.hidden_size_per_partition = (
            self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
        )
        self.query_key_value = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size * 3,
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
            gather_output=False,
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
            input_is_parallel=True,
        )
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.query_key_value(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(
            bsz,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).transpose(1, 2)
        k = k.view(
            bsz,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).transpose(1, 2)
        v = v.view(
            bsz,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (
            self.hidden_size_per_attention_head ** -0.5
        )
        scores = scores.masked_fill(
            ~expand_key_padding_mask(attention_mask, batch_size=bsz, seq_len=seq_len),
            torch.finfo(scores.dtype).min,
        )
        scores = scores.masked_fill(
            ~causal_mask(seq_len, device=x.device),
            torch.finfo(scores.dtype).min,
        )
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.hidden_size_per_partition
        )
        out = self.dense(context)
        return out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)


class MegatronMLP(nn.Module):
    activation_name = "gelu"

    def __init__(self, config: MegatronConfig) -> None:
        super().__init__()
        intermediate = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else int(config.hidden_size) * 4
        )
        self.dense_h_to_4h = ColumnParallelLinear(
            int(config.hidden_size),
            intermediate,
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
            gather_output=False,
        )
        self.dense_4h_to_h = RowParallelLinear(
            intermediate,
            int(config.hidden_size),
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
            input_is_parallel=True,
        )
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense_h_to_4h(x))
        x = self.dropout(x)
        return self.dense_4h_to_h(x)


class MegatronBlock(nn.Module):
    def __init__(self, config: MegatronConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.attention = MegatronAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.mlp = MegatronMLP(config)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.input_layernorm(x), attention_mask))
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        return x


class MegatronModel(nn.Module):
    def __init__(self, config: MegatronConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = VocabParallelEmbedding(
            int(config.vocab_size),
            int(config.hidden_size),
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
        )
        self.layers = nn.ModuleList([MegatronBlock(config) for _ in range(int(config.num_layers))])
        self.final_layernorm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.lm_head = VocabParallelLMHead(
            self.word_embeddings.weight,
            vocab_size=int(config.vocab_size),
            tensor_model_parallel_size=int(config.tensor_model_parallel_size),
            tensor_model_parallel_rank=int(config.tensor_model_parallel_rank),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.word_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layernorm(x)
        return self.lm_head(x)


__all__ = [
    "ColumnParallelLinear",
    "MegatronAttention",
    "MegatronBlock",
    "MegatronConfig",
    "MegatronMLP",
    "MegatronModel",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelLMHead",
]
