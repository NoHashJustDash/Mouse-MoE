# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end training, resume and export on synthetic shards (marker: torch)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_model import TINY
from test_teacher import _write_labels

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")

from mouse_moe.config import (  # noqa: E402
    ConfigError,
    DataConfig,
    LossConfig,
    RunConfig,
    TrainConfig,
)
from mouse_moe.export_onnx import export  # noqa: E402
from mouse_moe.label_teacher import shards_to_label  # noqa: E402
from mouse_moe.model import MouseMoe, RoutedMoe, expand_planes  # noqa: E402
from mouse_moe.score_teacher import score_checkpoint, teacher_validation_batches  # noqa: E402
from mouse_moe.teacher import TeacherLabelError  # noqa: E402
from mouse_moe.train import (  # noqa: E402
    calibrate_balance,
    initial_weights,
    lr_factor,
    train,
    validate,
)


def _config(data_dir: Path) -> RunConfig:
    return RunConfig(
        name="smoke",
        model=TINY,
        data=DataConfig(
            directory=str(data_dir),
            train_shards=("shard-0.tar", "shard-1.tar"),
            validation_shards=("shard-2.tar",),
            validation_positions_per_game=8,
            workers=1,
            readers_per_worker=2,
            shuffle_buffer=32,
            sample_rate=1.0,
        ),
        loss=LossConfig(),
        train=TrainConfig(
            batch_size=16,
            grad_accum_steps=2,  # micro-batches of 8
            steps=6,
            warmup_steps=2,
            lr_boundaries=(4,),
            lr_factors=(0.1,),
            precision="fp32",
            swa_start=4,
            swa_every=1,
            validate_every=3,
            validation_batches=2,
            checkpoint_every=3,
            log_every=1,
            balance_calibration_batches=5,
        ),
    )


def test_train_resume_and_export(synthetic_shards: Path, tmp_path: Path) -> None:
    config = _config(synthetic_shards)
    run = tmp_path / "run"
    device = torch.device("cpu")
    train(config, run, device, resume=False, max_steps=3)
    assert (run / "checkpoints" / "step_0000003.pt").is_file()
    with pytest.raises(FileExistsError):
        train(config, run, device, resume=False, max_steps=1)
    train(config, run, device, resume=True, max_steps=None)
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 6 and state["swa_started"] and state["swa_calibrated"]

    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert {e["event"] for e in events} >= {
        "start",
        "train",
        "validation",
        "balance_calibrated",
        "stop",
    }
    assert all("val_policy" in e for e in events if e["event"] == "validation")

    changed = replace(config, train=replace(config.train, lr=1e-3))
    with pytest.raises(ValueError, match="config differs"):
        train(changed, run, device, resume=True, max_steps=1)
    # Loader worker count may change on resume (memory only; see RESUME_MUTABLE_DATA).
    fewer_workers = replace(config, data=replace(config.data, workers=2))
    train(fewer_workers, run, device, resume=True, max_steps=1)

    report = export(run / "checkpoints" / "latest.pt", tmp_path / "net.onnx", True, 16)
    assert (tmp_path / "net.onnx").is_file() and report["max_abs_error_policy"] < 1e-3


def _expert_share(model: MouseMoe, planes: "torch.Tensor") -> "torch.Tensor":
    """Largest expert's share of the routing decisions, per layer."""
    model.train()
    with torch.no_grad():
        model(planes)
    shares = []
    for layer in model.moe_layers():
        assert isinstance(layer, RoutedMoe)
        shares.append(layer.load.max() / layer.load.sum())
        layer.load.zero_()
    return torch.stack(shares)


def test_balance_calibration_rebalances_without_touching_weights() -> None:
    torch.manual_seed(0)
    model = MouseMoe(TINY)
    for layer in model.moe_layers():
        assert isinstance(layer, RoutedMoe)
        layer.balance_bias.copy_(torch.tensor([0.1, 0.0, 0.0, 0.0]))  # expert 0 hogs tokens
    generator = torch.Generator().manual_seed(1)

    def batches():  # type: ignore[no-untyped-def]
        while True:
            yield {
                "packed": torch.randint(0, 256, (16, 832), dtype=torch.uint8, generator=generator),
                "aux": torch.rand(16, 8, generator=generator),
            }

    probe = next(batches())
    planes = expand_planes(probe["packed"], probe["aux"])
    before = _expert_share(model, planes)
    weights = {k: v.clone() for k, v in model.state_dict().items() if "balance_bias" not in k}
    calibrate_balance(model, batches(), 200, torch.device("cpu"), autocast=False)
    after = _expert_share(model, planes)
    assert (before > 0.45).all() and (after < before - 0.1).all()
    for key, value in model.state_dict().items():
        if "balance_bias" not in key:
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_resume_loads_the_checkpoint_to_the_cpu(
    synthetic_shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: map_location=device kept a second copy of model, optimizer and SWA
    # state on the GPU for the whole run; resuming ab_m1_fixed ran out of memory.
    config = _config(synthetic_shards)
    run = tmp_path / "run"
    train(config, run, torch.device("cpu"), resume=False, max_steps=3)
    seen: list[object] = []
    real_load = torch.load

    def recording_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    train(config, run, torch.device("cpu"), resume=True, max_steps=1)
    assert seen == ["cpu"]


def test_init_from_starts_a_fresh_run_at_the_source_swa_weights(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    source = _config(synthetic_shards)
    train(source, tmp_path / "source", torch.device("cpu"), resume=False, max_steps=None)
    checkpoint = tmp_path / "source" / "checkpoints" / "latest.pt"
    state = torch.load(checkpoint, weights_only=False)
    swa = {k.removeprefix("module."): v for k, v in state["swa"].items() if k != "n_averaged"}
    assert any(not torch.equal(swa[k], v) for k, v in state["model"].items())
    assert initial_weights(state).keys() == state["model"].keys()

    # A new schedule and fewer shards, as a game-target run fine-tuned on teacher labels.
    tuned = replace(
        source,
        name="tuned",
        data=replace(source.data, train_shards=("shard-0.tar",)),
        train=replace(source.train, steps=4, lr=1e-4, swa_start=2),
    )
    run = tmp_path / "tuned"
    with pytest.raises(ValueError, match="exclusive"):
        train(
            tuned, run, torch.device("cpu"), False, None, fork_from=checkpoint, init_from=checkpoint
        )
    seen: dict[str, torch.Tensor] = {}
    original = MouseMoe.forward

    def first_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not seen:
            seen.update({k: v.detach().clone() for k, v in self.state_dict().items()})
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MouseMoe, "forward", first_forward)
        train(tuned, run, torch.device("cpu"), False, None, init_from=checkpoint)
    for key, value in swa.items():  # trained from the averaged weights, not the live ones
        torch.testing.assert_close(seen[key], value, rtol=0, atol=0)
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [e["source_step"] for e in events if e["event"] == "initialized"] == [6]
    assert torch.load(run / "checkpoints" / "latest.pt", weights_only=False)["step"] == 4

    wider = replace(tuned, name="wider", model=replace(TINY, d_model=32))
    with pytest.raises(ValueError, match="different model config"):
        train(wider, tmp_path / "wider", torch.device("cpu"), False, None, init_from=checkpoint)


def test_labeler_takes_the_validation_shard_first_only_when_asked(synthetic_shards: Path) -> None:
    config = _config(synthetic_shards)
    names = [path.name for path in shards_to_label(config, include_validation=False)]
    assert names == ["shard-0.tar", "shard-1.tar"]
    names = [path.name for path in shards_to_label(config, include_validation=True)]
    assert names == ["shard-2.tar", "shard-0.tar", "shard-1.tar"]


def test_score_teacher_scores_the_played_net_on_teacher_targets(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    plain = _config(synthetic_shards)
    config = replace(plain, data=replace(plain.data, teacher_labels=str(labels)))
    device = torch.device("cpu")
    train(config, tmp_path / "run", device, resume=False, max_steps=None)
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"

    batches = teacher_validation_batches(config)
    assert len(batches) == 4  # 2 x 16 positions in micro-batches of 8
    for batch in batches:
        torch.testing.assert_close(batch["wdl"], torch.tensor([[0.25, 0.5, 0.25]]).expand(8, 3))

    report = score_checkpoint(checkpoint, batches, device)
    assert report["step"] == 6 and report["swa"] is True
    state = torch.load(checkpoint, weights_only=False)
    played = MouseMoe(TINY)
    played.load_state_dict(initial_weights(state))
    expected = validate(played, batches, config, device, len(batches))
    assert report["val_policy"] == pytest.approx(expected["val_policy"])
    live = MouseMoe(TINY)
    live.load_state_dict(state["model"])
    assert validate(live, batches, config, device, len(batches))["val_policy"] != pytest.approx(
        report["val_policy"]
    )

    with pytest.raises(ConfigError, match="teacher_labels"):
        teacher_validation_batches(plain)
    empty = replace(plain, data=replace(plain.data, teacher_labels=str(tmp_path / "none")))
    with pytest.raises(TeacherLabelError, match="no complete teacher labels"):
        teacher_validation_batches(empty)


def test_lr_rewarm_ramps_from_its_start_factor_to_the_schedule(synthetic_shards: Path) -> None:
    base = _config(synthetic_shards)
    config = replace(
        base,
        train=replace(
            base.train,
            lr_boundaries=(4, 6, 8),
            lr_factors=(0.1, 0.4, 0.2),
            rewarm_start=6,
            rewarm_steps=2,
            rewarm_from=0.1,
        ),
    )
    factors = [lr_factor(step, config) for step in range(10)]
    assert factors[4:] == pytest.approx([0.1, 0.1, 0.25, 0.4, 0.2, 0.2])
    assert [lr_factor(step, base) for step in range(10)] == pytest.approx(
        [0.5, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )


def test_fork_keeps_the_optimizer_and_restarts_swa_in_a_later_window(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    source = _config(synthetic_shards)  # 6 steps, SWA from step 4
    train(source, tmp_path / "source", torch.device("cpu"), resume=False, max_steps=None)
    checkpoint = tmp_path / "source" / "checkpoints" / "latest.pt"
    parent = torch.load(checkpoint, weights_only=False)
    assert parent["swa_started"] and int(parent["swa"]["n_averaged"]) == 3

    stage2 = replace(
        source,
        name="stage2",
        data=replace(source.data, teacher_weight=1.0),
        train=replace(
            source.train,
            steps=10,
            lr_boundaries=(4, 6, 8),
            lr_factors=(0.1, 0.4, 0.2),
            rewarm_start=6,
            rewarm_steps=2,
            rewarm_from=0.1,
            swa_start=8,
        ),
    )
    run = tmp_path / "stage2"
    train(stage2, run, torch.device("cpu"), False, None, fork_from=checkpoint)
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 10 and state["swa_started"] and state["swa_calibrated"]
    # SWA restarted: only steps 8, 9 and 10 are in the average, none of the parent's.
    assert int(state["swa"]["n_averaged"]) == 3
    # The optimizer continued: Adam-family step counts carry on from the parent's 6.
    steps = {int(s["step"]) for s in state["optimizer"]["state"].values()}
    assert steps == {10}

    late = replace(stage2, name="late", train=replace(stage2.train, swa_start=5))
    with pytest.raises(ValueError, match="cannot be separated"):
        train(late, tmp_path / "late", torch.device("cpu"), False, None, fork_from=checkpoint)
