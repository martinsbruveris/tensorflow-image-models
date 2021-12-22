import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tfimm.train import config
from tfimm.train.registry import cfg_serializable


@dataclasses.dataclass
class SimpleConfig:
    i: int
    f: float
    b: bool
    s: str
    t: tuple

    id: int = 0
    fd: float = 1.0
    bd: bool = True
    sd: str = "abc"
    td: tuple = (1, 2)


@cfg_serializable
class SimpleClass:
    cfg_class = SimpleConfig


@dataclasses.dataclass
class OuterConfig:
    n: Any
    n_class: str
    cfg_file: str = ""


def test_parse_simple_class():
    cfg = SimpleConfig(i=0, f=0.0, b=False, s="", t=tuple())
    cmdline_args = [
        "--i=2",
        "--f=2.5",
        "--b=t",
        "--s=d",
        "--t=(1, 1)",
        "--sd=",
        "--td=()",
    ]
    expected = SimpleConfig(i=2, f=2.5, b=True, s="d", t=(1, 1), sd="", td=tuple())
    cfg = config.parse_args(cfg, args=cmdline_args)
    assert cfg == expected


@pytest.mark.parametrize("outer_as_class", [True, False])
@pytest.mark.parametrize("n_class", ["SimpleClass", "", None])
def test_parse_nested(outer_as_class, n_class):
    """Parse a nested config with either a known type for the sub-config or an unknwon
    type. The outer config is either a dict or a dataclass."""
    if outer_as_class:
        cfg = OuterConfig(n=None, n_class=n_class)
    else:
        # Note that key `n` is not present here.
        cfg = {"n_class": n_class}

    cmdline_args = ["--n_class=SimpleClass"]
    cmdline_args += ["--n.i=2", "--n.f=2.5", "--n.b=f", "--n.s=d", "--n.t=(1, 1)"]
    cfg = config.parse_args(cfg, args=cmdline_args)

    expected = {
        "n": SimpleConfig(i=2, f=2.5, b=False, s="d", t=(1, 1)),
        "n_class": "SimpleClass",
    }
    if outer_as_class:
        expected = OuterConfig(**expected)

    assert cfg == expected


def test_flat_to_deep():
    """Tests conversion between flat and deep configs."""
    nested_cfg = {
        "problem": {"nb_classes": 10},
        "problem_class": "ClassificationProblem",
    }
    flat_cfg = {
        "problem.nb_classes": 10,
        "problem_class": "ClassificationProblem",
    }
    assert flat_cfg == config.deep_to_flat(nested_cfg)
    assert nested_cfg == config.flat_to_deep(flat_cfg)


def test_empty_nesting():
    cfg = {"n_class": ""}
    cmdline_args = ["--n_class="]
    cfg = config.parse_args(cfg, args=cmdline_args)

    expected = {"n": None, "n_class": ""}
    assert cfg == expected


def test_config_file():
    """Test parsing a config from a yaml file."""
    cfg_yaml = """
    n:
      i: 2
      f: 2.5
      b: false
      s: d
      t: !!python/tuple [1, 1]
    n_class: SimpleClass
    """
    cfg_file = tempfile.NamedTemporaryFile()
    Path(cfg_file.name).write_text(cfg_yaml)
    cfg = {"cfg_file": cfg_file.name}

    cfg = config.parse_args(cfg, cfg_class=OuterConfig, args=[])

    expected = OuterConfig(
        n=SimpleConfig(i=2, f=2.5, b=False, s="d", t=(1, 1)),
        n_class="SimpleClass",
        cfg_file=cfg_file.name,
    )

    assert cfg == expected
    cfg_file.close()
