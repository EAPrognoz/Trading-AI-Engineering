from __future__ import annotations

from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest

from trading_ai.data.mt5_adapter import fetch_h1_bars


class FakeMT5:
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self) -> None:
        self.shutdown_called = False
        self.calls = []

    def initialize(self) -> bool:
        return True

    def last_error(self):
        return (1, "synthetic failure")

    def copy_rates_range(self, *args, **kwargs):
        self.calls.append(args)
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


@pytest.mark.parametrize(
    ("timeframe", "mt5_constant"),
    [("H1", 60), ("H4", 240), ("D1", 1440)],
)
def test_fetch_market_bars_maps_h1_h4_d1(
    monkeypatch, timeframe: str, mt5_constant: int
) -> None:
    from trading_ai.data.mt5_adapter import fetch_market_bars

    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="source failure"):
        fetch_market_bars("BTCUSD", timeframe, start, end)

    assert fake.calls == [("BTCUSD", mt5_constant, start, end)]
    assert fake.shutdown_called


def test_fetch_market_bars_rejects_unsupported_before_terminal_use(monkeypatch) -> None:
    from trading_ai.data.mt5_adapter import fetch_market_bars

    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        fetch_market_bars("BTCUSD", "M15", start, end)
    assert fake.calls == []


def test_successful_fetch_records_terminal_build_and_package_version(monkeypatch) -> None:
    from trading_ai.data.mt5_adapter import fetch_market_bars

    fake = FakeMT5()
    fake.__version__ = "5.0-test"
    fake.copy_rates_range = lambda *args: [{"time": 1767571200, "open": 1.0}]
    fake.terminal_info = lambda: SimpleNamespace(build=5432)
    fake.version = lambda: (5, 0, 5432)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)
    frame = fetch_market_bars("BTCUSD", "H1", start, end)

    assert frame.loc[0, "timestamp"].tzinfo is not None
    assert frame.attrs["mt5_provenance"]["terminal_build"] == 5432
    assert frame.attrs["mt5_provenance"]["terminal_version"] == [5, 0, 5432]
    assert frame.attrs["mt5_provenance"]["metatrader5_package_version"] == "5.0-test"
    assert fake.shutdown_called


def test_fetch_rejects_missing_terminal_build(monkeypatch) -> None:
    from trading_ai.data.mt5_adapter import fetch_market_bars

    fake = FakeMT5()
    fake.__version__ = "5.0-test"
    fake.copy_rates_range = lambda *args: [{"time": 1767571200, "open": 1.0}]
    fake.terminal_info = lambda: None
    fake.version = lambda: (5, 0, 5432)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="terminal build"):
        fetch_market_bars("BTCUSD", "H1", start, end)
    assert fake.shutdown_called
