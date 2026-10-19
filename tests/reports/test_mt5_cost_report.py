from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import sys

import pytest

from trading_ai.reports.mt5_cost_report import (
    AccountScope,
    DealValidationError,
    HistoryFetchError,
    build_report,
    fetch_demo_history,
    require_history_result,
    write_report_bundle,
)


UTC = timezone.utc
START = datetime(2026, 10, 17, tzinfo=UTC)
END = datetime(2026, 10, 18, tzinfo=UTC)
SCOPE = AccountScope(
    login="SYNTHETIC-DEMO-ACCOUNT",
    server="SYNTHETIC-NOT-A-BROKER",
    currency="USD",
)


def deal(
    ticket: int,
    when: datetime,
    *,
    type: int = 0,
    entry: int = 0,
    profit: float = 0.0,
    commission: float = 0.0,
    swap: float = 0.0,
    fee: float = 0.0,
    **extra,
):
    epoch_ms = int(when.timestamp() * 1000)
    return {
        "ticket": ticket,
        "order": ticket + 1000,
        "time": epoch_ms // 1000,
        "time_msc": epoch_ms,
        "type": type,
        "entry": entry,
        "position_id": ticket + 2000,
        "symbol": "SYNTHETIC_EURUSD" if type in (0, 1) else "",
        "profit": profit,
        "commission": commission,
        "swap": swap,
        "fee": fee,
        "magic": 0,
        "reason": 0,
        "external_id": "",
        "volume": 0.1 if type in (0, 1) else 0.0,
        "price": 1.1 if type in (0, 1) else 0.0,
        **extra,
    }


def test_daily_subtotal_keeps_entry_commissions_and_components_separate():
    deals = [
        deal(1, START, type=0, entry=0, commission=-2.5, fee=-0.1),
        deal(
            2,
            START + timedelta(hours=1),
            type=1,
            entry=1,
            profit=10.0,
            commission=-2.5,
            swap=0.25,
            fee=-0.1,
        ),
    ]

    report = build_report(
        deals, start=START, end=END, account_scope=SCOPE
    )

    day = report.buy_sell_daily[0]
    assert day["deal_count"] == 2
    assert day["profit"] == "10.0"
    assert day["commission"] == "-5.0"
    assert day["swap"] == "0.25"
    assert day["fee"] == "-0.2"
    assert day["buy_sell_recorded_component_subtotal"] == "5.05"
    assert report.raw_deals[0]["commission"] == -2.5
    assert report.raw_deals[0]["entry"] == 0
    assert report.reconciliation["account_wide_net_profit_claim"] is False


def test_balance_deposit_withdrawal_and_unknown_type_remain_reviewable():
    deals = [
        deal(10, START, type=2, profit=100.0),
        deal(11, START + timedelta(hours=1), type=2, profit=-25.0),
        deal(12, START + timedelta(hours=2), type=99, profit=0.2, commission=-0.3),
    ]

    report = build_report(deals, start=START, end=END, account_scope=SCOPE)

    assert report.buy_sell_daily == ()
    assert [row["ticket"] for row in report.other_operations] == [10, 11, 12]
    assert [row["profit"] for row in report.other_operations] == [100.0, -25.0, 0.2]
    assert report.other_operations[2]["deal_type_label"] == "UNKNOWN_TYPE_99"
    assert "no allocation applied" in report.other_operations[2]["review_reason"]
    assert report.reconciliation["other_operation_count"] == 3
    assert report.reconciliation["unknown_type_count"] == 1


def test_none_error_is_distinct_from_valid_empty_history():
    with pytest.raises(HistoryFetchError, match="last_error=.*-10004"):
        require_history_result(None, (-10004, "terminal unavailable"))
    assert require_history_result((), (1, "no error")) == ()

    empty = build_report((), start=START, end=END, account_scope=SCOPE)
    assert empty.reconciliation["result_status"] == "empty_valid_response"
    assert empty.reconciliation["source_record_count"] == 0
    assert empty.reconciliation["all_source_records_partitioned_exactly_once"]
    assert empty.buy_sell_daily == ()
    assert empty.other_operations == ()


def test_half_open_interval_accepts_start_and_excludes_end_but_keeps_source_rows():
    deals = [
        deal(20, START - timedelta(milliseconds=1), profit=50.0),
        deal(21, START, profit=1.0),
        deal(22, END - timedelta(milliseconds=1), profit=2.0),
        deal(23, END, profit=100.0),
    ]

    report = build_report(deals, start=START, end=END, account_scope=SCOPE)

    assert [row["report_partition"] for row in report.raw_deals] == [
        "outside_requested_interval",
        "buy_sell_deal",
        "buy_sell_deal",
        "outside_requested_interval",
    ]
    assert report.reconciliation["partition_counts"] == {
        "buy_sell_deal": 2,
        "other_operation": 0,
        "outside_requested_interval": 2,
    }
    assert report.reconciliation["source_record_count"] == 4
    assert report.reconciliation["in_requested_interval_record_count"] == 2
    assert report.buy_sell_daily[0]["profit"] == "3.0"
    assert report.reconciliation["requested_interval"]["policy"] == "[start, end)"


def test_timestamp_conversion_and_interval_provenance_keep_microsecond_precision():
    submillisecond_start = START + timedelta(microseconds=500)
    submillisecond_end = START + timedelta(microseconds=1500)
    source_timestamp = START + timedelta(milliseconds=1)
    report = build_report(
        [deal(24, source_timestamp)],
        start=submillisecond_start,
        end=submillisecond_end,
        account_scope=SCOPE,
    )

    assert report.raw_deals[0]["time_utc"] == "2026-10-17T00:00:00.001Z"
    interval = report.reconciliation["requested_interval"]
    assert interval["start_utc_inclusive"] == "2026-10-17T00:00:00.000500Z"
    assert interval["end_utc_exclusive"] == "2026-10-17T00:00:00.001500Z"
    assert report.reconciliation["in_requested_interval_record_count"] == 1


def test_swap_signs_are_preserved_and_netted_as_recorded():
    deals = [
        deal(30, START, swap=0.75),
        deal(31, START + timedelta(minutes=1), swap=-0.4),
    ]
    report = build_report(deals, start=START, end=END, account_scope=SCOPE)
    assert [row["swap"] for row in report.raw_deals] == [0.75, -0.4]
    assert report.buy_sell_daily[0]["swap"] == "0.35"


def test_duplicate_ticket_is_rejected_visibly():
    repeated = deal(40, START)
    with pytest.raises(DealValidationError, match="duplicate deal ticket '40'"):
        build_report(
            [repeated, dict(repeated)], start=START, end=END, account_scope=SCOPE
        )


def test_duplicate_ticket_is_rejected_across_integral_numeric_representations():
    first = deal(41, START)
    second = deal(41.0, START + timedelta(seconds=1))
    with pytest.raises(DealValidationError, match="duplicate deal ticket '41'"):
        build_report([first, second], start=START, end=END, account_scope=SCOPE)


@pytest.mark.parametrize("field", ["profit", "commission", "swap", "fee"])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_money_is_rejected(field, invalid):
    row = deal(50, START)
    row[field] = invalid
    with pytest.raises(DealValidationError, match=f"{field}="):
        build_report([row], start=START, end=END, account_scope=SCOPE)


@pytest.mark.parametrize(
    "scope_field,wrong_value,expected_text",
    [
        ("account_login", "OTHER-DEMO-ACCOUNT", "account_login"),
        ("account_currency", "EUR", "account_currency"),
        ("account_server", "OTHER-DEMO-SERVER", "account_server"),
    ],
)
def test_source_rows_cannot_cross_account_or_currency_scope(
    scope_field, wrong_value, expected_text
):
    with pytest.raises(DealValidationError, match=expected_text):
        build_report(
            [deal(60, START, **{scope_field: wrong_value})],
            start=START,
            end=END,
            account_scope=SCOPE,
        )


def test_real_account_is_rejected_before_history_request(monkeypatch):
    class FakeMT5:
        ACCOUNT_TRADE_MODE_DEMO = 0

        def __init__(self):
            self.history_requested = False
            self.shutdown_called = False

        def initialize(self):
            return True

        def account_info(self):
            return {
                "login": 123,
                "server": "Broker-Real",
                "currency": "USD",
                "trade_mode": 2,
            }

        def history_deals_get(self, *args):
            self.history_requested = True
            return ()

        def last_error(self):
            return (0, "ok")

        def shutdown(self):
            self.shutdown_called = True

    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    with pytest.raises(RuntimeError, match="not verified as demo"):
        fetch_demo_history(START, END)
    assert fake.history_requested is False
    assert fake.shutdown_called is True


def test_demo_fetch_uses_one_scoped_account_and_shuts_down(monkeypatch):
    expected = [deal(70, START)]

    class FakeMT5:
        ACCOUNT_TRADE_MODE_DEMO = 0

        def __init__(self):
            self.shutdown_called = False
            self.request = None

        def initialize(self):
            return True

        def account_info(self):
            return {
                "login": 7001,
                "server": "Synthetic-Demo-Server",
                "currency": "USD",
                "trade_mode": 0,
            }

        def history_deals_get(self, start, end):
            self.request = (start, end)
            return expected

        def last_error(self):
            return (0, "ok")

        def shutdown(self):
            self.shutdown_called = True

    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    deals, scope = fetch_demo_history(
        START,
        END,
        expected_login="7001",
        expected_server="Synthetic-Demo-Server",
    )
    assert deals == tuple(expected)
    assert scope.currency == "USD"
    assert scope.login == "7001"
    assert fake.request == (START, END)
    assert fake.shutdown_called is True


@pytest.mark.parametrize(
    "history_result,expect_error",
    [(None, True), ((), False)],
    ids=["none-is-error", "empty-is-valid"],
)
def test_terminal_none_and_empty_results_have_distinct_paths_and_shutdown(
    monkeypatch, history_result, expect_error
):
    class FakeMT5:
        ACCOUNT_TRADE_MODE_DEMO = 0

        def __init__(self):
            self.shutdown_called = False
            self.last_error_calls = 0

        def initialize(self):
            return True

        def account_info(self):
            return {
                "login": 7002,
                "server": "Synthetic-Demo-Server",
                "currency": "USD",
                "trade_mode": 0,
            }

        def history_deals_get(self, start, end):
            return history_result

        def last_error(self):
            self.last_error_calls += 1
            return (-10004, "synthetic terminal failure")

        def shutdown(self):
            self.shutdown_called = True

    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    if expect_error:
        with pytest.raises(HistoryFetchError, match="synthetic terminal failure"):
            fetch_demo_history(START, END)
        assert fake.last_error_calls == 1
    else:
        deals, _scope = fetch_demo_history(START, END)
        assert deals == ()
        assert fake.last_error_calls == 0
    assert fake.shutdown_called is True


def test_writer_creates_four_named_outputs_and_preserves_existing_files(tmp_path):
    report = build_report(
        [deal(80, START, profit=1.0)], start=START, end=END, account_scope=SCOPE
    )
    output = tmp_path / "report"
    paths = write_report_bundle(report, output)
    assert set(paths) == {
        "raw_deals.csv",
        "buy_sell_daily.csv",
        "other_operations.csv",
        "reconciliation.json",
    }
    assert all(path.is_file() for path in paths.values())
    with paths["raw_deals.csv"].open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1

    sentinel = tmp_path / "existing" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        write_report_bundle(report, sentinel.parent)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
