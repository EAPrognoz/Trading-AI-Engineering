from __future__ import annotations

import pytest

from trading_ai.data.mt5_adapter import fetch_h1_bars
from trading_ai.data.request import MarketDataRequest


class FakeMT5:
    TIMEFRAME_H1 = 60

    def __init__(self) -> None:
        self.shutdown_called = False

    def initialize(self) -> bool:
        return True

    def last_error(self):
        return (1, "synthetic failure")

    def symbol_select(self, symbol: str, enabled: bool) -> bool:
        return True

    def copy_rates_range(self, *args, **kwargs):
        raise RuntimeError("source failure")

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_adapter_always_shuts_down_when_fetch_fails() -> None:
    fake = FakeMT5()
    request = MarketDataRequest(
        symbol="EURUSD",
        start="2026-01-05T08:00:00Z",
        end="2026-01-05T12:00:00Z",
        cutoff="2026-01-05T12:00:00Z",
    )

    with pytest.raises(RuntimeError, match="source failure"):
        fetch_h1_bars(request, mt5_module=fake)

    assert fake.shutdown_called is True
