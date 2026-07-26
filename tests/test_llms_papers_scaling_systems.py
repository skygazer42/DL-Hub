import pytest

torch = pytest.importorskip("torch")


def test_megatron_uses_tensor_parallel_layers_and_vocab_partitioning() -> None:
    from Llms.megatron import MegatronConfig, MegatronModel

    model = MegatronModel(
        MegatronConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_attention_heads=4,
            num_layers=2,
            intermediate_size=64,
            tensor_model_parallel_size=2,
            tensor_model_parallel_rank=0,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert block.attention.num_attention_heads_per_partition == 2
    assert block.attention.query_key_value.is_column_parallel is True
    assert block.mlp.dense_h_to_4h.is_column_parallel is True
    assert block.mlp.dense_4h_to_h.is_row_parallel is True
    assert block.mlp.activation_name == "gelu"
    assert model.word_embeddings.is_vocab_parallel is True
    assert model.word_embeddings.vocab_start_index == 0
    assert model.word_embeddings.vocab_end_index == 32
    assert model.word_embeddings.weight.shape[0] == 32
    assert model.word_embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_megatron_parallel_layers_can_load_and_reconstruct_dense_weights() -> None:
    import torch.nn.functional as F
    from Llms.megatron import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

    full_col_weight = torch.randn(12, 8)
    full_col_bias = torch.randn(12)
    col0 = ColumnParallelLinear(
        8, 12, tensor_model_parallel_size=2, tensor_model_parallel_rank=0, bias=True
    )
    col1 = ColumnParallelLinear(
        8, 12, tensor_model_parallel_size=2, tensor_model_parallel_rank=1, bias=True
    )
    col0.load_full_parameters(full_col_weight, full_col_bias)
    col1.load_full_parameters(full_col_weight, full_col_bias)
    x = torch.randn(2, 3, 8)
    dense_col = F.linear(x, full_col_weight, full_col_bias)
    parted_col = torch.cat((col0(x), col1(x)), dim=-1)
    gathered_col_weight, gathered_col_bias = ColumnParallelLinear.gather_full_parameters(
        (col0, col1)
    )
    assert torch.allclose(parted_col, dense_col, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_col_weight, full_col_weight)
    assert torch.allclose(gathered_col_bias, full_col_bias)

    full_row_weight = torch.randn(10, 8)
    full_row_bias = torch.randn(10)
    row0 = RowParallelLinear(
        8, 10, tensor_model_parallel_size=2, tensor_model_parallel_rank=0, bias=True
    )
    row1 = RowParallelLinear(
        8, 10, tensor_model_parallel_size=2, tensor_model_parallel_rank=1, bias=True
    )
    row0.load_full_parameters(full_row_weight, full_row_bias)
    row1.load_full_parameters(full_row_weight, full_row_bias)
    dense_row = F.linear(x, full_row_weight, full_row_bias)
    parted_row = row0.forward_partial(x) + row1.forward_partial(x) + full_row_bias
    gathered_row_weight, gathered_row_bias = RowParallelLinear.gather_full_parameters((row0, row1))
    assert torch.allclose(parted_row, dense_row, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_row_weight, full_row_weight)
    assert torch.allclose(gathered_row_bias, full_row_bias)

    full_embed_weight = torch.randn(16, 6)
    emb0 = VocabParallelEmbedding(
        16, 6, tensor_model_parallel_size=2, tensor_model_parallel_rank=0
    )
    emb1 = VocabParallelEmbedding(
        16, 6, tensor_model_parallel_size=2, tensor_model_parallel_rank=1
    )
    emb0.load_full_weight(full_embed_weight)
    emb1.load_full_weight(full_embed_weight)
    token_ids = torch.tensor([[0, 5, 8, 15]], dtype=torch.long)
    dense_embed = F.embedding(token_ids, full_embed_weight)
    parted_embed = emb0(token_ids) + emb1(token_ids)
    gathered_embed = VocabParallelEmbedding.gather_full_weight((emb0, emb1))
    assert torch.allclose(parted_embed, dense_embed, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_embed, full_embed_weight)

def test_chinchilla_recommends_more_tokens_for_same_compute_budget() -> None:
    from Llms.chinchilla import ChinchillaConfig, ChinchillaPlanner

    planner = ChinchillaPlanner(ChinchillaConfig())
    canonical = planner.plan_for_parameters(70_000_000_000)
    gopher_budget = planner.training_flops(
        parameters=280_000_000_000,
        tokens=300_000_000_000,
    )
    optimal = planner.plan_for_compute(gopher_budget)

    assert canonical.tokens == 1_400_000_000_000
    assert canonical.tokens_per_parameter == pytest.approx(20.0)
    assert optimal.parameters < 280_000_000_000
    assert optimal.tokens > 300_000_000_000
    assert optimal.compute_budget_flops == pytest.approx(gopher_budget)

def test_gpipe_balances_cells_builds_pipeline_schedule_and_matches_sequential() -> None:
    from Llms.gpipe import GPipeConfig, GPipeSequential

    layers = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ]
    )
    reference = torch.nn.Sequential(*layers)
    pipeline = GPipeSequential(
        layers,
        GPipeConfig(num_partitions=3, micro_batches=4, rematerialization=True),
    )
    x = torch.randn(8, 4)
    schedule = pipeline.pipeline_schedule()

    with torch.no_grad():
        expected = reference(x)
        actual = pipeline(x)

    assert pipeline.partition_sizes == [2, 2, 1]
    assert pipeline.rematerialization is True
    assert pipeline.bubble_steps == 2
    assert len(schedule) == 6
    assert schedule[0] == ((0, 0),)
    assert schedule[-1] == ((2, 3),)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

def test_pathways_maps_virtual_devices_gang_schedules_and_traces_fused_programs() -> None:
    from Llms.pathways import PathwaysConfig, PathwaysProgram, PathwaysRuntime, PathwaysTracer, VirtualDevice

    devices = [
        VirtualDevice(logical_id="v2", island="island-b", physical_device="pod-b:7"),
        VirtualDevice(logical_id="v0", island="island-a", physical_device="pod-a:3"),
        VirtualDevice(logical_id="v1", island="island-b", physical_device="pod-b:1"),
        VirtualDevice(logical_id="v3", island="island-a", physical_device="pod-a:8"),
    ]
    runtime = PathwaysRuntime(PathwaysConfig(interleave_quantum=1), devices)
    tracer = PathwaysTracer()
    tracer.add_compiled("embed")
    tracer.add_compiled("dispatch")
    tracer.add_compiled("decode")
    fused = tracer.fuse("serve", required_devices=2)
    retrieval = PathwaysProgram(name="retrieve", stages=("lookup", "merge"), required_devices=2)

    mapped = runtime.map_virtual_devices(fused)
    gang = runtime.gang_schedule([fused, retrieval])
    interleaved = runtime.interleave([fused, retrieval])

    assert [device.logical_id for device in mapped] == ["v0", "v1"]
    assert fused.compiled_functions == ("embed", "dispatch", "decode")
    assert fused.stages == ("embed", "dispatch", "decode")
    assert gang[0].program.name == "serve"
    assert len(gang[0].devices) == 2
    assert gang[0].islands == ("island-a", "island-b")
    assert [step.program_name for step in interleaved[:4]] == ["serve", "retrieve", "serve", "retrieve"]

def test_zero_partitions_states_by_stage_and_reconstructs_parameters() -> None:
    from Llms.zero import ZeROConfig, ZeROEngine

    parameter = torch.arange(12, dtype=torch.float32)
    gradient = parameter + 0.5
    optimizer_state = {
        "momentum": parameter + 1.0,
        "variance": parameter + 2.0,
    }

    stage1 = ZeROEngine(ZeROConfig(stage=1, world_size=3, rank=1))
    shard1 = stage1.partition_states(
        parameter=parameter,
        gradient=gradient,
        optimizer_state=optimizer_state,
    )
    assert shard1.plan.partitions_optimizer_state is True
    assert shard1.plan.partitions_gradients is False
    assert shard1.plan.partitions_parameters is False
    assert shard1.parameter.numel() == 12
    assert shard1.gradient.numel() == 12
    assert all(value.numel() == 4 for value in shard1.optimizer_state.values())

    stage2 = ZeROEngine(ZeROConfig(stage=2, world_size=3, rank=1))
    shard2 = stage2.partition_states(
        parameter=parameter,
        gradient=gradient,
        optimizer_state=optimizer_state,
    )
    assert shard2.gradient.numel() == 4
    assert shard2.parameter.numel() == 12

    stage3_shards = [
        ZeROEngine(ZeROConfig(stage=3, world_size=3, rank=rank)).partition_states(
            parameter=parameter,
            gradient=gradient,
            optimizer_state=optimizer_state,
        )
        for rank in range(3)
    ]
    assert all(shard.parameter.numel() == 4 for shard in stage3_shards)
    reconstructed = stage3_shards[0].engine.gather_parameters(
        [shard.parameter for shard in stage3_shards]
    )
    assert torch.allclose(reconstructed, parameter, atol=1e-6, rtol=1e-6)

def test_parameter_server_supports_sparse_pull_push_and_ssp_progress() -> None:
    from Llms.parameter_server import ParameterServer, ParameterServerConfig

    server = ParameterServer(
        {"embedding": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        ParameterServerConfig(num_workers=2, consistency="ssp", staleness=1),
    )
    worker0 = server.register_worker("worker0")
    worker1 = server.register_worker("worker1")

    pulled = worker0.pull("embedding", indices=torch.tensor([1, 3], dtype=torch.long))
    worker0.push(
        "embedding",
        indices=torch.tensor([1, 3], dtype=torch.long),
        values=torch.tensor([0.5, -1.0]),
    )

    assert torch.allclose(pulled, torch.tensor([2.0, 4.0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        server.parameters["embedding"],
        torch.tensor([1.0, 2.5, 3.0, 3.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert worker0.finish_step() is True
    assert worker0.finish_step() is True
    assert worker0.finish_step() is False
    assert worker1.finish_step() is True
    assert worker0.finish_step() is True
