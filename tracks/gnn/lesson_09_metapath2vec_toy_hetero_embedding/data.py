from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataConfig:
    # Graph size
    num_authors: int = 24
    num_papers: int = 48
    num_venues: int = 6

    # Edge structure
    authors_per_paper: int = 2
    papers_per_author: int = 4

    # Random walks
    metapath: str = "A2P,P2A"
    start_type: str = "author"
    num_walks: int = 200
    walk_length: int = 30
    window_size: int = 5

    # Negative sampling
    care_type: int = 0  # 0: global negatives, 1: same-type negatives
    seed: int = 0


@dataclass(frozen=True)
class ToyHeteroGraph:
    num_nodes: int
    type_names: tuple[str, ...]
    node_types: torch.Tensor  # (N,)
    node_names: list[str]
    rel_neighbors: dict[str, list[torch.Tensor]]
    rel_src_type: dict[str, int]
    rel_dst_type: dict[str, int]


def _neighbors_from_edges(
    num_nodes: int, src: torch.Tensor, dst: torch.Tensor
) -> list[torch.Tensor]:
    buckets: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for s, d in zip(src.tolist(), dst.tolist()):
        buckets[int(s)].append(int(d))
    out: list[torch.Tensor] = []
    for b in buckets:
        if b:
            out.append(torch.tensor(b, dtype=torch.long))
        else:
            out.append(torch.empty((0,), dtype=torch.long))
    return out


def build_toy_hetero_graph(cfg: DataConfig) -> ToyHeteroGraph:
    """Create a small heterogeneous graph with (author, paper, venue) types.

    Relations (directed):
    - A2P / P2A: authorship edges
    - P2V / V2P: publication venue edges
    """

    g = torch.Generator().manual_seed(int(cfg.seed))

    type_names = ("author", "paper", "venue")
    n_a, n_p, n_v = int(cfg.num_authors), int(cfg.num_papers), int(cfg.num_venues)
    off_a, off_p, off_v = 0, n_a, n_a + n_p
    num_nodes = n_a + n_p + n_v

    node_types = torch.empty((num_nodes,), dtype=torch.long)
    node_types[off_a : off_a + n_a] = 0
    node_types[off_p : off_p + n_p] = 1
    node_types[off_v : off_v + n_v] = 2

    node_names: list[str] = []
    node_names += [f"author_{i}" for i in range(n_a)]
    node_names += [f"paper_{i}" for i in range(n_p)]
    node_names += [f"venue_{i}" for i in range(n_v)]

    # ---- A <-> P edges (authorship) ----
    # Ensure every author is connected to a few papers.
    a2p_src: list[int] = []
    a2p_dst: list[int] = []

    papers = torch.arange(n_p, dtype=torch.long)
    for a in range(n_a):
        chosen = papers[torch.randperm(n_p, generator=g)[: int(cfg.papers_per_author)]]
        for p in chosen.tolist():
            a2p_src.append(off_a + a)
            a2p_dst.append(off_p + int(p))

    # Ensure every paper has at least `authors_per_paper` authors.
    authors = torch.arange(n_a, dtype=torch.long)
    for p in range(n_p):
        chosen = authors[torch.randperm(n_a, generator=g)[: int(cfg.authors_per_paper)]]
        for a in chosen.tolist():
            a2p_src.append(off_a + int(a))
            a2p_dst.append(off_p + p)

    a2p_src_t = torch.tensor(a2p_src, dtype=torch.long)
    a2p_dst_t = torch.tensor(a2p_dst, dtype=torch.long)
    p2a_src_t = a2p_dst_t.clone()
    p2a_dst_t = a2p_src_t.clone()

    # ---- P <-> V edges (venue) ----
    venue_ids = torch.randint(low=0, high=n_v, size=(n_p,), generator=g, dtype=torch.long)
    p2v_src_t = torch.arange(off_p, off_p + n_p, dtype=torch.long)
    p2v_dst_t = off_v + venue_ids
    v2p_src_t = p2v_dst_t.clone()
    v2p_dst_t = p2v_src_t.clone()

    rel_src_type = {"A2P": 0, "P2A": 1, "P2V": 1, "V2P": 2}
    rel_dst_type = {"A2P": 1, "P2A": 0, "P2V": 2, "V2P": 1}

    rel_neighbors = {
        "A2P": _neighbors_from_edges(num_nodes, a2p_src_t, a2p_dst_t),
        "P2A": _neighbors_from_edges(num_nodes, p2a_src_t, p2a_dst_t),
        "P2V": _neighbors_from_edges(num_nodes, p2v_src_t, p2v_dst_t),
        "V2P": _neighbors_from_edges(num_nodes, v2p_src_t, v2p_dst_t),
    }

    return ToyHeteroGraph(
        num_nodes=num_nodes,
        type_names=type_names,
        node_types=node_types,
        node_names=node_names,
        rel_neighbors=rel_neighbors,
        rel_src_type=rel_src_type,
        rel_dst_type=rel_dst_type,
    )


def parse_metapath(metapath: str) -> tuple[str, ...]:
    rels = [p.strip() for p in metapath.split(",") if p.strip()]
    if not rels:
        raise ValueError("metapath must be a non-empty comma-separated relation list, e.g. A2P,P2A")
    return tuple(rels)


def generate_walks(graph: ToyHeteroGraph, cfg: DataConfig) -> torch.Tensor:
    metapath = parse_metapath(cfg.metapath)
    for rel in metapath:
        if rel not in graph.rel_neighbors:
            raise ValueError(
                f"Unknown relation in metapath: {rel}. Known: {sorted(graph.rel_neighbors.keys())}"
            )

    if cfg.start_type not in graph.type_names:
        raise ValueError(f"Unknown start_type={cfg.start_type!r}. Known: {graph.type_names}")

    start_type_id = graph.type_names.index(cfg.start_type)
    if graph.rel_src_type[metapath[0]] != start_type_id:
        raise ValueError(
            f"metapath starts with {metapath[0]} (src type {graph.type_names[graph.rel_src_type[metapath[0]]]}) "
            f"but start_type is {cfg.start_type}"
        )

    g = torch.Generator().manual_seed(int(cfg.seed))
    start_nodes = torch.nonzero(graph.node_types == start_type_id, as_tuple=False).view(-1)
    if int(start_nodes.numel()) == 0:
        raise RuntimeError("No start nodes found for start_type")

    walks = torch.empty((int(cfg.num_walks), int(cfg.walk_length) + 1), dtype=torch.long)
    for w in range(int(cfg.num_walks)):
        cur = int(start_nodes[torch.randint(0, int(start_nodes.numel()), (1,), generator=g)].item())
        walks[w, 0] = cur
        for t in range(int(cfg.walk_length)):
            rel = metapath[t % len(metapath)]
            neigh = graph.rel_neighbors[rel][cur]
            if int(neigh.numel()) == 0:
                # Restart at a random node of this relation's src type.
                src_type = graph.rel_src_type[rel]
                candidates = torch.nonzero(graph.node_types == src_type, as_tuple=False).view(-1)
                cur = int(
                    candidates[torch.randint(0, int(candidates.numel()), (1,), generator=g)].item()
                )
                neigh = graph.rel_neighbors[rel][cur]
            nxt = (
                cur
                if int(neigh.numel()) == 0
                else int(neigh[torch.randint(0, int(neigh.numel()), (1,), generator=g)].item())
            )
            walks[w, t + 1] = nxt
            cur = nxt
    return walks


def build_skipgram_pairs(
    walks: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(window_size) < 1:
        raise ValueError("window_size must be >= 1")

    walks = walks.to(torch.long)
    num_walks, length = walks.shape

    centers: list[int] = []
    contexts: list[int] = []
    for w in range(int(num_walks)):
        seq = walks[w].tolist()
        for i in range(int(length)):
            c = int(seq[i])
            lo = max(0, i - int(window_size))
            hi = min(int(length), i + int(window_size) + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                centers.append(c)
                contexts.append(int(seq[j]))

    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


class SkipGramPairs(Dataset):
    def __init__(self, centers: torch.Tensor, contexts: torch.Tensor) -> None:
        if centers.shape != contexts.shape:
            raise ValueError("centers and contexts must have the same shape")
        self.centers = centers.to(torch.long)
        self.contexts = contexts.to(torch.long)

    def __len__(self) -> int:
        return int(self.centers.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.centers[idx], self.contexts[idx]


class NegativeSampler:
    def __init__(self, *, node_types: torch.Tensor, seed: int, care_type: int) -> None:
        self.node_types = node_types.to(torch.long).cpu()
        self.care_type = int(care_type)
        self.gen = torch.Generator().manual_seed(int(seed))

        self.num_nodes = int(self.node_types.numel())
        self.nodes_by_type: dict[int, torch.Tensor] = {}
        for t in torch.unique(self.node_types).tolist():
            nodes = torch.nonzero(self.node_types == int(t), as_tuple=False).view(-1)
            self.nodes_by_type[int(t)] = nodes.to(torch.long).cpu()

    def sample(self, context: torch.Tensor, k: int) -> torch.Tensor:
        context = context.to(torch.long).cpu()
        b = int(context.numel())
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")

        if self.care_type == 0:
            return torch.randint(
                low=0, high=self.num_nodes, size=(b, k), generator=self.gen, dtype=torch.long
            )

        # Same-type negatives (metapath2vec++ style): sample negatives with the same type as the context node.
        ctx_types = self.node_types[context]  # (B,)
        out = torch.empty((b, k), dtype=torch.long)
        for t in torch.unique(ctx_types).tolist():
            t = int(t)
            mask = ctx_types == t
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            candidates = self.nodes_by_type[t]
            if int(candidates.numel()) == 0:
                out[idx] = torch.randint(
                    low=0, high=self.num_nodes, size=(int(idx.numel()), k), generator=self.gen
                )
                continue
            sampled = candidates[
                torch.randint(
                    low=0,
                    high=int(candidates.numel()),
                    size=(int(idx.numel()), k),
                    generator=self.gen,
                )
            ]
            out[idx] = sampled
        return out


def build_training_pairs(cfg: DataConfig) -> tuple[ToyHeteroGraph, SkipGramPairs, NegativeSampler]:
    graph = build_toy_hetero_graph(cfg)
    walks = generate_walks(graph, cfg)
    centers, contexts = build_skipgram_pairs(walks, window_size=cfg.window_size)
    ds = SkipGramPairs(centers, contexts)
    sampler = NegativeSampler(
        node_types=graph.node_types, seed=cfg.seed + 1, care_type=cfg.care_type
    )
    return graph, ds, sampler


__all__ = [
    "DataConfig",
    "ToyHeteroGraph",
    "build_toy_hetero_graph",
    "parse_metapath",
    "generate_walks",
    "build_skipgram_pairs",
    "SkipGramPairs",
    "NegativeSampler",
    "build_training_pairs",
]
