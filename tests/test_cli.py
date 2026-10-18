"""Tests for the qriton-hlm CLI (no checkpoint needed)."""

import sys

import pytest

import qriton_hlm
from qriton_hlm import cli


def test_version_flag_reports_package_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qriton-hlm", "--version"])
    with pytest.raises(SystemExit):
        cli.main()
    assert capsys.readouterr().out.strip() == f"qriton-hlm {qriton_hlm.__version__}"
