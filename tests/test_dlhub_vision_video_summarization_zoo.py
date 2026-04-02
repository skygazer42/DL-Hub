import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in video summarization zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_video_summarization_zoo_lists_families() -> None:
    from dlhub.vision.video_summarization_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 99
    assert "vsum:dsn_tiny" in arches
    assert "vsum:sum_gan_small" in arches
    assert "vsum:cycle_sum_base" in arches
    assert "vsum:vasnet_tiny" in arches
    assert "vsum:dsnet_small" in arches
    assert "vsum:ca_sum_tiny" in arches
    assert "vsum:pgl_sum_tiny" in arches
    assert "vsum:mhscnet_small" in arches
    assert "vsum:tac_sum_base" in arches
    assert "vsum:csta_tiny" in arches
    assert "vsum:fulltransnet_small" in arches
    assert "vsum:summdiff_base" in arches
    assert "vsum:qfvs_memnet_tiny" in arches
    assert "vsum:videograph_small" in arches
    assert "vsum:lgrln_base" in arches
    assert "vsum:intentvizor_tiny" in arches
    assert "vsum:maam_small" in arches
    assert "vsum:checkmate_base" in arches
    assert "vsum:viewpoint_sum_tiny" in arches
    assert "vsum:progressive_ssl_small" in arches
    assert "vsum:llm_pretrain_base" in arches
    assert "vsum:contrast_sum_tiny" in arches
    assert "vsum:mc_vsa_small" in arches
    assert "vsum:multi_stream_sum_base" in arches
    assert "vsum:personalized_ranker_tiny" in arches
    assert "vsum:sem_reward_rl_small" in arches
    assert "vsum:dp_dtw_sum_base" in arches
    assert "vsum:hsa_rnn_tiny" in arches
    assert "vsum:clip_it_small" in arches
    assert "vsum:videosage_base" in arches
    assert "vsum:pfmn_tiny" in arches
    assert "vsum:a2summ_small" in arches
    assert "vsum:iterative_gan_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "vsum:dsn_tiny",
        "vsum:sum_gan_tiny",
        "vsum:cycle_sum_tiny",
        "vsum:vasnet_tiny",
        "vsum:dsnet_tiny",
        "vsum:ca_sum_tiny",
        "vsum:pgl_sum_tiny",
        "vsum:mhscnet_tiny",
        "vsum:tac_sum_tiny",
        "vsum:csta_tiny",
        "vsum:fulltransnet_tiny",
        "vsum:summdiff_tiny",
        "vsum:qfvs_memnet_tiny",
        "vsum:videograph_tiny",
        "vsum:lgrln_tiny",
        "vsum:intentvizor_tiny",
        "vsum:maam_tiny",
        "vsum:checkmate_tiny",
        "vsum:viewpoint_sum_tiny",
        "vsum:progressive_ssl_tiny",
        "vsum:llm_pretrain_tiny",
        "vsum:contrast_sum_tiny",
        "vsum:mc_vsa_tiny",
        "vsum:multi_stream_sum_tiny",
        "vsum:personalized_ranker_tiny",
        "vsum:sem_reward_rl_tiny",
        "vsum:dp_dtw_sum_tiny",
        "vsum:hsa_rnn_tiny",
        "vsum:clip_it_tiny",
        "vsum:videosage_tiny",
        "vsum:pfmn_tiny",
        "vsum:a2summ_tiny",
        "vsum:iterative_gan_tiny",
    ],
)
def test_video_summarization_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.vision.video_summarization_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        seq_len=8,
        image_size=32,
        width_mult=0.5,
        dropout=0.0,
    )

    video = torch.randn(2, 8, 3, 32, 32)
    out = model(video)
    assert isinstance(out, dict)
    assert "scores" in out
    assert "summary_mask" in out
    assert tuple(out["scores"].shape) == (2, 8)
    assert tuple(out["summary_mask"].shape) == (2, 8)

    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


def test_video_summarization_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/video_summarization_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Video summarization local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/video_summarization_zoo.py", "--smoke", "vsum:vasnet_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: vsum:vasnet_tiny" in smoke_proc.stdout
