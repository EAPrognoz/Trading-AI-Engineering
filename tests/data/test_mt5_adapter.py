from __future__ import annotations

from datetime import datetime, timezone
import sys

import pytest

from trading_ai.data.mt5_adapter import fetch_h1_bars


class FakeMT5:
    TIMEFRAME_H1 = 60

    def __init__(self) -> None:
        self.shutdown_called = False

    def initialize(self) -> bool:
        return True

    def last_error(self):
        return (1, "synthetic failure")

    def copy_rates_range(self, *args, **kwargs):
        raise RuntimeError("source failure")

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_adapter_always_shuts_down_when_fetch_fails(monkeypatch) -> None:
    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="source failure"):
        fetch_h1_bars("EURUSD", start, end)

    assert fake.shutdown_called is True
