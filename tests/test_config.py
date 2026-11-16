# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import ROOT
from mouse_moe.budget import parameter_total
from mouse_moe.config import ConfigError, load_config, load_config_dict, to_dict, validate

CONFIGS = sorted((ROOT / "configs").rglob("*.toml"))


def test_configs_exist() -> None:
    names = {path.stem for path in CONFIGS}
    assert {
        "m1_46m",
        "ab_d0",
        "ab_d1",
        "ab_m1",
        "ab_m2",
        "ab_m3",
        "ab_m4",
        "ab_m5",
        "depth_l8",
        "depth_l12",
        "dense_full_control",
    } <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_is_valid_and_within_budget(path: Path) -> None:
    config = load_config(path)
    assert parameter_total(config.model) <= config.model.parameter_budget
    assert load_config_dict(to_dict(config)) == config


def test_full_model_budget() -> None:
    config = load_config(ROOT / "configs" / "m1_46m.toml")
    assert parameter_total(config.model) == 46_044_976


def test_rejects_unknown_keys_and_inconsistency(tmp_path: Path) -> None:
    good = load_config(ROOT / "configs" / "ablation" / "ab_m1.toml")
    raw = to_dict(good)
    raw["model"]["typo"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        load_config_dict(raw)
    overlap = replace(good, data=replace(good.data, validation_shards=good.data.train_shards[:1]))
    with pytest.raises(ConfigError, match="both"):
        validate(overlap)
    half_game_source = replace(good, data=replace(good.data, game_shards=("x.tar",)))
    with pytest.raises(ConfigError, match="game_directory"):
        validate(half_game_source)
    clash = replace(
        good,
        data=replace(good.data, game_directory="d", game_shards=good.data.train_shards[:1]),
    )
    with pytest.raises(ConfigError, match="share a name"):
        validate(clash)
    orphan_labels = replace(good, data=replace(good.data, game_teacher_labels="labels"))
    with pytest.raises(ConfigError, match="game_teacher_labels"):
        validate(orphan_labels)
    bad_heads = replace(good, model=replace(good.model, heads=7))
    with pytest.raises(ConfigError, match="divisible"):
        validate(bad_heads)
