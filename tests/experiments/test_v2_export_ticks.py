import csv
from datetime import datetime, timezone
import json
import sys

import pytest

from examples import eurusd_v2_export_ticks as exporter


class FakeMT5:
    COPY_TICKS_INFO = 2

    def __init__(self, ticks):
        self.ticks = ticks
        self.initialized = False
        self.shutdown_called = False
        self.request = None

    def initialize(self):
        self.initialized = True
        return True

    def copy_ticks_range(self, symbol, start, end, flags):
        self.request = (symbol, start, end, flags)
        return self.ticks

    def last_error(self):
        return (-10004, "synthetic terminal error")

    def shutdown(self):
        self.shutdown_called = True


def _tick(time_msc, bid="1.10000", ask="1.10020", flags=1):
    return {"time_msc":time_msc,"bid":bid,"ask":ask,"flags":flags}


def test_exporter_keeps_duplicate_timestamps_and_filters_exclusive_end(tmp_path, monkeypatch):
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    start_ms = int(start.timestamp()) * 1000
    fake = FakeMT5((_tick(start_ms), _tick(start_ms), _tick(start_ms + 120_000)))
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    csv_path, manifest_path = exporter.export(
        "EURUSD", start, datetime(2026, 1, 5, 0, 2, tzinfo=timezone.utc), tmp_path / "ticks.csv"
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    manifest = json.loads(manifest_path.read_text())
    assert [row["source_row_seq"] for row in rows] == ["0", "1"]
    assert rows[0]["time_msc"] == rows[1]["time_msc"]
    assert manifest["terminal_response_rows"] == 3
    assert manifest["saved_rows"] == 2
    assert manifest["timestamps_deduplicated"] is False
    assert fake.request[3] == FakeMT5.COPY_TICKS_INFO
    assert fake.shutdown_called


def test_exporter_distinguishes_none_error_from_valid_empty(tmp_path, monkeypatch):
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    empty = FakeMT5(())
    monkeypatch.setitem(sys.modules, "MetaTrader5", empty)
    csv_path, manifest_path = exporter.export(
        "EURUSD", start, datetime(2026, 1, 6, tzinfo=timezone.utc), tmp_path / "empty.csv"
    )
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["time_msc,bid,ask,flags,source_row_seq"]
    assert json.loads(manifest_path.read_text())["status"] == "valid_empty_response"
    assert empty.shutdown_called

    failed = FakeMT5(None)
    monkeypatch.setitem(sys.modules, "MetaTrader5", failed)
    with pytest.raises(RuntimeError, match="synthetic terminal error"):
        exporter.export(
            "EURUSD", start, datetime(2026, 1, 6, tzinfo=timezone.utc), tmp_path / "failed.csv"
        )
    assert failed.shutdown_called
    assert not (tmp_path / "failed.csv").exists()


def test_exporter_rejects_backwards_source_order_and_shuts_down(tmp_path, monkeypatch):
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    start_ms = int(start.timestamp()) * 1000
    fake = FakeMT5((_tick(start_ms + 1_000), _tick(start_ms)))
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    with pytest.raises(ValueError, match="moved backwards"):
        exporter.export(
            "EURUSD", start, datetime(2026, 1, 6, tzinfo=timezone.utc), tmp_path / "bad.csv"
        )
    assert fake.shutdown_called
