"""
Requires kraken (run under .venv-kraken, not the shared/base env):
    .venv-kraken/Scripts/python.exe -m pytest tests/test_kraken_infer.py
"""

import pytest

kraken = pytest.importorskip("kraken")

from PIL import Image

from src.kraken.infer import clean_output, direct_line_segmentation


def test_direct_line_segmentation_stays_in_bounds():
    # Regression test: kraken.lib.segmentation requires every polygon/
    # baseline coordinate to be strictly < (width, height) — using width/
    # height themselves (an off-by-one) raises "Line polygon outside of
    # image bounds" at recognition time. Caught by hand against a real
    # model; this pins the fix so it can't silently regress.
    img = Image.new("RGB", (200, 50))
    seg = direct_line_segmentation(img)

    line = seg.lines[0]
    xs = [p[0] for p in line.boundary] + [p[0] for p in line.baseline]
    ys = [p[1] for p in line.boundary] + [p[1] for p in line.baseline]

    assert max(xs) < img.width
    assert max(ys) < img.height
    assert min(xs) >= 0
    assert min(ys) >= 0


def test_direct_line_segmentation_is_baseline_type():
    # The CATMuS model warns "will likely result in severely degraded
    # performance" (and empirically does — verified against real
    # predictions) when given a bbox-type segmentation instead of the
    # baseline type it was trained on.
    img = Image.new("RGB", (300, 80))
    seg = direct_line_segmentation(img)
    assert seg.type == "baselines"
    assert seg.lines[0].type == "baselines"


def test_direct_line_segmentation_covers_whole_image():
    img = Image.new("RGB", (300, 80))
    seg = direct_line_segmentation(img)
    boundary_xs = [p[0] for p in seg.lines[0].boundary]
    boundary_ys = [p[1] for p in seg.lines[0].boundary]
    assert min(boundary_xs) == 0
    assert min(boundary_ys) == 0
    assert max(boundary_xs) == img.width - 1
    assert max(boundary_ys) == img.height - 1


def test_clean_output_strips_role_tags():
    assert clean_output("user\ntranscribe\nassistant\nBy this Act") == "By this Act"


def test_clean_output_collapses_whitespace():
    assert clean_output("By  this\n\nAct") == "By this Act"
