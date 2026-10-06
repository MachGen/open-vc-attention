"""CLI smoke checks, not accepted performance measurements."""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.large,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="V repair requires B200",
    ),
    pytest.mark.skipif(
        os.environ.get("OPEN_VC_ATTN_TEST_LARGE") != "1", reason="Set OPEN_VC_ATTN_TEST_LARGE=1"
    ),
]


@pytest.mark.parametrize("scope", ["attention", "quantize-attention"])
@pytest.mark.parametrize("timer", ["events", "graph"])
def test_repair_cli_scope_and_timer(tmp_path, scope, timer):
    target = tmp_path / "repair.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_vc_attn.cli.bench",
            "--shapes",
            "32769x2x128",
            "--backends",
            "bf16",
            "open-vc",
            "--scope",
            scope,
            "--timing",
            timer,
            "--repair-budget",
            "0.005",
            "--rounds",
            "2",
            "--repeats",
            "1",
            "--warm-calls",
            "1",
            "--warm-seconds",
            "0",
            "--allow-shared-gpu",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(target.read_text())
    assert report["status"] == "complete"
    assert report["scope"] == scope and report["timing"] == timer
    assert report["isolation_checked"] is False
    candidate = report["shapes"][0]
    metadata = candidate["backend_metadata"]["open-vc"]
    assert metadata["repair_budget"] == 0.005
    assert metadata["selected_tokens_per_head"] == 164
    assert metadata["repair_tokens_per_head"] == 256
    assert candidate["accuracy"]["open-vc"]["finite"]
    assert set(candidate["timings"]) == {"bf16", "open-vc"}
    assert "Shared-GPU result" in result.stdout
