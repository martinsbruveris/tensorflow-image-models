import dataclasses

from tfimm.train.config import parse_args
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


def test_parse_simple_class():
    cfg = SimpleConfig(i=0, f=0.0, b=False, s="", t=tuple())
    cmdline_args = [
        "--i=2", "--f=2.5", "--b=t", "--s=d", "--t=(1, 1)", "--sd=", "--td=()"
    ]
    expected = SimpleConfig(i=2, f=2.5, b=True, s="d", t=(1, 1), sd="", td=tuple())
    cfg = parse_args(cfg, cmdline_args)
    assert cfg == expected


def test_parse_nested():
    cfg = {"n_class": "SimpleClass"}
    cmdline_args = ["--n.i=2", "--n.f=2.5", "--n.b=f", "--n.s=d", "--n.t=(1, 1)"]
    expected = {
        "n": SimpleConfig(i=2, f=2.5, b=False, s="d", t=(1, 1)),
        "n_class": "SimpleClass",
    }
    cfg = parse_args(cfg, cmdline_args)
    assert cfg == expected


def test_parse_unknown():
    cfg = {"n_class": dataclasses.MISSING}
    cmdline_args = ["--n_class=SimpleClass"]
    cmdline_args += ["--n.i=2", "--n.f=2.5", "--n.b=f", "--n.s=d", "--n.t=(1, 1)"]
    expected = {
        "n": SimpleConfig(i=2, f=2.5, b=False, s="d", t=(1, 1)),
        "n_class": "SimpleClass",
    }
    cfg = parse_args(cfg, cmdline_args)
    assert cfg == expected
