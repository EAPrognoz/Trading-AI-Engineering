import csv
from datetime import datetime, timedelta, timezone
import json

import pytest

from experiments.eurusd_h1_model_error_lab.backtest.engine import ReplayInputError
from experiments.eurusd_h1_model_error_lab.backtest.v2.replay_csv import run


def _fixture_files(tmp_path):
    """Small synthetic rows for software behavior only, never market evidence."""
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    decisions = tmp_path / "synthetic_decisions.csv"
    with decisions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signal_id", "decision_time_utc", "action"])
        writer.writeheader()
        writer.writerows([
            {"signal_id":"open-long","decision_time_utc":"2026-01-05T00:00:00Z","action":"long"},
            {"signal_id":"same-side","decision_time_utc":"2026-01-05T00:30:00Z","action":"long"},
            {"signal_id":"abstain","decision_time_utc":"2026-01-05T00:45:00Z","action":"ABSTAIN"},
        ])
    ticks = tmp_path / "synthetic_ticks.csv"
    offsets = (0, 1, 1800, 2700, 3600, 3601)
    with ticks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_msc", "bid", "ask", "flags", "source_row_seq"])
        writer.writeheader()
        for seq, offset in enumerate(offsets):
            instant = start + timedelta(seconds=offset)
            epoch_ms = int(instant.timestamp()) * 1000
            writer.writerow({"time_msc":epoch_ms,"bid":"1.10000","ask":"1.10020","flags":1,"source_row_seq":seq})
    return start, decisions, ticks


def test_csv_runner_preserves_same_side_and_abstain_policy_and_writes_manifest(tmp_path):
    start, decisions, ticks = _fixture_files(tmp_path)
    output = run(decisions, ticks, start, start + timedelta(hours=2), tmp_path / "run")

    positions = list(csv.DictReader((output / "positions.csv").open(encoding="utf-8")))
    actions = {row["signal_id"]: row["disposition"] for row in csv.DictReader((output / "signal_actions.csv").open(encoding="utf-8"))}
    manifest = json.loads((output / "run_manifest.json").read_text())

    assert len(positions) == 1
    assert positions[0]["signal_id"] == "open-long"
    assert positions[0]["target_time_utc"] == "2026-01-05T01:00:00.000Z"
    assert positions[0]["exit_reason"] == "target"
    assert actions["same-side"] == "same_side_noop"
    assert actions["abstain"] == "abstain_hold"
    assert manifest["scenario"]["same_side_policy"] == "keep_original_target"
    assert manifest["scenario"]["abstain_policy"] == "hold"
    assert manifest["read_only"] is True
    assert manifest["account_wide_net_profit_claim"] is False
    assert manifest["trading_profitability_test"] is False
    assert manifest["inputs"]["tick_rows"] == 6
    with pytest.raises(ReplayInputError, match="already exists"):
        run(decisions, ticks, start, start + timedelta(hours=2), output)


def test_csv_runner_rejects_end_boundary_quote(tmp_path):
    start, decisions, ticks = _fixture_files(tmp_path)
    rows = list(csv.DictReader(ticks.open(encoding="utf-8")))
    rows[-1]["time_msc"] = str(int((start + timedelta(hours=2)).timestamp()) * 1000)
    with ticks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ReplayInputError, match=r"outside \[start, end\)"):
        run(decisions, ticks, start, start + timedelta(hours=2), tmp_path / "invalid")


def test_csv_runner_rejects_nonfinite_quote_value(tmp_path):
    start, decisions, ticks = _fixture_files(tmp_path)
    rows = list(csv.DictReader(ticks.open(encoding="utf-8")))
    rows[1]["bid"] = "NaN"
    with ticks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ReplayInputError):
        run(decisions, ticks, start, start + timedelta(hours=2), tmp_path / "invalid")
