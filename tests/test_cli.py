"""Tests for the qriton-hlm CLI (no checkpoint needed)."""

import sys

import pytest
import torch

import qriton_hlm
from qriton_hlm import cli


def test_version_flag_reports_package_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qriton-hlm", "--version"])
    with pytest.raises(SystemExit):
        cli.main()
    assert capsys.readouterr().out.strip() == f"qriton-hlm {qriton_hlm.__version__}"


def test_inject_accepts_auto_strength(monkeypatch):
    monkeypatch.setattr(cli.C, "_enabled", False)
    torch.manual_seed(0)
    lang = cli.EnergyLang()
    assert "random W sandbox" in lang.execute("load __random_64")
    out = lang.execute("inject 0 42 auto")
    assert "Error" not in out and "BLOCKED" not in out
    assert "α=auto" in out
    # the sandbox W has no basins to protect, so the write is kept
    assert "After:" in out and "target is basin: True" in out
    assert "YES" in lang.execute("verify 0 42")
