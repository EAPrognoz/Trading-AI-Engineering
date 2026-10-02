from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trading_ai.data.request import MarketDataRequest


ANALYSIS_START = "2026-01-05T08:00:00Z"
ANALYSIS_END = "2026-01-05T12:00:00Z"


def _bars(opens: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(opens, utc=True),
            "open": [100.0] * len(opens),
            "high": [101.0] * len(opens),
            "low": [99.0] * len(opens),
            "close": [100.5] * len(opens),
            "tick_volume": [10] * len(opens),
            "spread": [1] * len(opens),
            "real_volume": [0] * len(opens),
        }
    )


def _create_streams(
    bundle_dir: Path, *, skip: str | None = None, reject: str | None = None,
    short: str | None = None, stale: str | None = None,
    tail_stale: str | None = None,
) -> None:
    from trading_ai.data.pipeline import process_market_response

    specs = {
        "H1": ("2026-01-05T06:00:00Z", "2026-01-05T12:00:00Z", [
            "2026-01-05T06:00:00Z", "2026-01-05T07:00:00Z",
            "2026-01-05T08:00:00Z", "2026-01-05T09:00:00Z",
            "2026-01-05T10:00:00Z", "2026-01-05T11:00:00Z",
        ]),
        "H4": ("2026-01-05T00:00:00Z", "2026-01-05T12:00:00Z", [
            "2026-01-05T00:00:00Z", "2026-01-05T04:00:00Z",
            "2026-01-05T08:00:00Z",
        ]),
        "D1": ("2026-01-03T01:00:00Z", ANALYSIS_END, [
            "2026-01-03T01:00:00Z", "2026-01-04T01:00:00Z",
        ]),
    }
    for timeframe, (start, end, opens) in specs.items():
        if timeframe == skip:
            continue
        if timeframe == short:
            opens = opens[1:]
            start = opens[0]
        if timeframe == stale:
            if timeframe == "H4":
                opens = [
                    "2026-01-04T16:00:00Z", "2026-01-04T20:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            elif timeframe == "D1":
                opens = [
                    "2026-01-01T01:00:00Z", "2026-01-02T01:00:00Z",
                    "2026-01-03T01:00:00Z",
                ]
            else:
                raise AssertionError("stale fixture supports H4 and D1 only")
            start = opens[0]
        if timeframe == tail_stale:
            if timeframe == "H4":
                opens = [
                    "2026-01-04T17:00:00Z", "2026-01-04T21:00:00Z",
                    "2026-01-05T01:00:00Z",
                ]
            elif timeframe == "D1":
                opens = [
                    "2026-01-01T09:00:00Z", "2026-01-02T09:00:00Z",
                    "2026-01-03T09:00:00Z",
                ]
            else:
                raise AssertionError("tail-stale fixture supports H4 and D1 only")
            start = opens[0]
        raw = _bars(opens)
        if timeframe == reject:
            raw.loc[1, "close"] = 200.0
        request = MarketDataRequest(
            symbol="BTCUSD.test", timeframe=timeframe,
            start=start, end=end, cutoff=ANALYSIS_END,
        )
        process_market_response(
            raw, request=request, run_dir=bundle_dir / timeframe,
            provenance={
                "source_type": "synthetic_fixture",
                "feature_max_lookback_bars": 1,
            },
            code_version="test",
        )


def _write_bundle(bundle_dir: Path) -> Path:
    from trading_ai.data.bundle import write_market_data_bundle_manifest

    write_market_data_bundle_manifest(
        bundle_dir, bundle_id="btc-synthetic-001",
        analysis_start=ANALYSIS_START, analysis_end=ANALYSIS_END,
    )
    return bundle_dir / "bundle_manifest.json"


def _write_hash_consistent_manifest_without_writer(bundle_dir: Path) -> Path:
    """Construct an otherwise valid manifest to exercise the loader gate alone."""
    from trading_ai.data.manifest import sha256_file

    members = {}
    for timeframe in ("H1", "H4", "D1"):
        source_path = bundle_dir / timeframe / "manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        accepted_path = bundle_dir / timeframe / "accepted.csv"
        accepted_sha = sha256_file(accepted_path)
        assert source["files"]["accepted_dataset"]["sha256"] == accepted_sha
        members[timeframe] = {
            "symbol": source["request"]["symbol"],
            "timeframe": timeframe,
            "dataset_id": source["dataset_id"],
            "source_manifest": {
                "path": f"{timeframe}/manifest.json",
                "sha256": sha256_file(source_path),
            },
            "accepted_dataset": {
                "path": f"{timeframe}/accepted.csv",
                "sha256": accepted_sha,
            },
        }
    path = bundle_dir / "bundle_manifest.json"
    path.write_text(json.dumps({
        "contract_id": "ep002-btc-multitimeframe-bundle-v1",
        "bundle_id": "btc-synthetic-001",
        "symbol": "BTCUSD.test",
        "analysis_start": pd.Timestamp(ANALYSIS_START).isoformat(),
        "analysis_end": pd.Timestamp(ANALYSIS_END).isoformat(),
        "analysis_range_policy": "start_inclusive_end_exclusive",
        "streams": members,
    }), encoding="utf-8")
    return path


def _assert_fresh_at_first_decision_but_stale_at_last(
    bundle_dir: Path, timeframe: str
) -> None:
    from trading_ai.data.timeframes import timeframe_duration

    frame = pd.read_csv(bundle_dir / timeframe / "accepted.csv")
    nominal_closes = pd.to_datetime(frame["timestamp"], utc=True) + timeframe_duration(timeframe)
    first = pd.Timestamp(ANALYSIS_START)
    last = pd.Timestamp(ANALYSIS_END) - pd.Timedelta(hours=1)
    latest_at_first = nominal_closes[nominal_closes <= first].iloc[-1]
    latest_at_last = nominal_closes[nominal_closes <= last].iloc[-1]
    assert latest_at_first > first - timeframe_duration(timeframe)
    assert latest_at_last <= last - timeframe_duration(timeframe)


@pytest.mark.parametrize("missing_or_rejected", ["missing", "rejected"])
def test_bundle_requires_all_three_accepted_streams(
    tmp_path: Path, missing_or_rejected: str
) -> None:
    bundle_dir = tmp_path / "bundle"
    _create_streams(
        bundle_dir,
        skip="D1" if missing_or_rejected == "missing" else None,
        reject="D1" if missing_or_rejected == "rejected" else None,
    )
    with pytest.raises(ValueError, match="D1"):
        _write_bundle(bundle_dir)
    assert not (bundle_dir / "bundle_manifest.json").exists()


def test_bundle_manifest_uses_relative_paths(tmp_path: Path) -> None:
    from trading_ai.data.bundle import load_market_data_bundle

    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir)
    path = _write_bundle(bundle_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contract_id"] == "ep002-btc-multitimeframe-bundle-v1"
    assert payload["bundle_id"] == "btc-synthetic-001"
    assert payload["symbol"] == "BTCUSD.test"
    for timeframe in ("H1", "H4", "D1"):
        member = payload["streams"][timeframe]
        assert member["timeframe"] == timeframe
        for key in ("source_manifest", "accepted_dataset"):
            member_path = Path(member[key]["path"])
            assert not member_path.is_absolute()
            assert member_path.parts[0] == timeframe
    loaded = load_market_data_bundle(path)
    assert set(loaded.frames) == {"H1", "H4", "D1"}
    assert loaded.manifest == payload
    assert len(loaded.manifest_sha256) == 64


@pytest.mark.parametrize("mutation", ["csv", "source_manifest", "symbol", "timeframe", "dataset_id", "absolute_path", "escape_path"])
def test_bundle_loader_rejects_hash_and_identity_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    from trading_ai.data.bundle import load_market_data_bundle

    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir)
    path = _write_bundle(bundle_dir)
    if mutation == "csv":
        with (bundle_dir / "H1" / "accepted.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif mutation == "source_manifest":
        with (bundle_dir / "H1" / "manifest.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        member = payload["streams"]["H1"]
        if mutation in {"symbol", "timeframe", "dataset_id"}:
            member[mutation] = "wrong"
        elif mutation == "absolute_path":
            member["accepted_dataset"]["path"] = str((bundle_dir / "H1" / "accepted.csv").resolve())
        else:
            member["accepted_dataset"]["path"] = "../outside.csv"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_market_data_bundle(path)


def test_bundle_manifest_never_overwrites_existing(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir)
    path = _write_bundle(bundle_dir)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _write_bundle(bundle_dir)
    assert path.read_bytes() == before


def test_bundle_rejects_inadequate_native_preroll(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir, short="H4")
    with pytest.raises(ValueError, match="H4.*pre-roll"):
        _write_bundle(bundle_dir)
    assert not (bundle_dir / "bundle_manifest.json").exists()


@pytest.mark.parametrize("timeframe", ["H4", "D1"])
def test_bundle_writer_rejects_continuous_but_stale_stream(
    tmp_path: Path, timeframe: str
) -> None:
    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir, stale=timeframe)
    with pytest.raises(ValueError, match=f"{timeframe}.*coverage"):
        _write_bundle(bundle_dir)
    assert not (bundle_dir / "bundle_manifest.json").exists()


@pytest.mark.parametrize("timeframe", ["H4", "D1"])
def test_bundle_loader_rejects_hash_consistent_but_stale_stream(
    tmp_path: Path, timeframe: str
) -> None:
    from trading_ai.data.bundle import load_market_data_bundle

    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir, stale=timeframe)
    path = _write_hash_consistent_manifest_without_writer(bundle_dir)
    with pytest.raises(ValueError, match=f"{timeframe}.*coverage"):
        load_market_data_bundle(path)


@pytest.mark.parametrize("timeframe", ["H4", "D1"])
def test_bundle_writer_rejects_tail_only_staleness(
    tmp_path: Path, timeframe: str
) -> None:
    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir, tail_stale=timeframe)
    _assert_fresh_at_first_decision_but_stale_at_last(bundle_dir, timeframe)
    with pytest.raises(ValueError, match=f"{timeframe}.*coverage"):
        _write_bundle(bundle_dir)
    assert not (bundle_dir / "bundle_manifest.json").exists()


@pytest.mark.parametrize("timeframe", ["H4", "D1"])
def test_bundle_loader_rejects_hash_consistent_tail_only_staleness(
    tmp_path: Path, timeframe: str
) -> None:
    from trading_ai.data.bundle import load_market_data_bundle

    bundle_dir = tmp_path / "bundle"
    _create_streams(bundle_dir, tail_stale=timeframe)
    _assert_fresh_at_first_decision_but_stale_at_last(bundle_dir, timeframe)
    path = _write_hash_consistent_manifest_without_writer(bundle_dir)
    with pytest.raises(ValueError, match=f"{timeframe}.*coverage"):
        load_market_data_bundle(path)


def test_bundle_cli_requires_symbol_and_unique_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.ep002_market_data import run_mt5_bundle

    contract = tmp_path / "contract.toml"
    contract.write_text(
        "[timeframes.H1]\nmax_lookback_bars = 24\n"
        "[timeframes.H4]\nmax_lookback_bars = 24\n"
        "[timeframes.D1]\nmax_lookback_bars = 24\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    called = []
    monkeypatch.setattr(run_mt5_bundle, "fetch_market_bars", lambda *a: called.append(a))
    common = [
        "--analysis-start", ANALYSIS_START, "--analysis-end", ANALYSIS_END,
        "--cutoff", ANALYSIS_END, "--feature-contract", str(contract),
        "--run-dir", str(run_dir),
    ]
    with pytest.raises(SystemExit):
        run_mt5_bundle.main(common)
    with pytest.raises(FileExistsError):
        run_mt5_bundle.main(["--symbol", "BTCUSD.test", *common])
    assert called == []


def test_bundle_cli_rejects_missing_or_invalid_lookback(tmp_path: Path) -> None:
    from experiments.ep002_market_data.run_mt5_bundle import _feature_lookbacks

    contract = tmp_path / "contract.toml"
    contract.write_text(
        "[timeframes.H1]\nmax_lookback_bars = 2\n"
        "[timeframes.H4]\nmax_lookback_bars = 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="H1/H4/D1"):
        _feature_lookbacks(contract)
    contract.write_text(
        "[timeframes.H1]\nmax_lookback_bars = 2\n"
        "[timeframes.H4]\nmax_lookback_bars = 0\n"
        "[timeframes.D1]\nmax_lookback_bars = 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="H4"):
        _feature_lookbacks(contract)


def test_bundle_cli_uses_native_preroll_and_writes_only_complete_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.ep002_market_data import run_mt5_bundle
    from trading_ai.data.bundle import load_market_data_bundle
    from trading_ai.data.manifest import sha256_file
    from trading_ai.data.timeframes import timeframe_duration

    contract = tmp_path / "contract.toml"
    contract.write_text(
        "[timeframes.H1]\nmax_lookback_bars = 2\n"
        "[timeframes.H4]\nmax_lookback_bars = 1\n"
        "[timeframes.D1]\nmax_lookback_bars = 1\n",
        encoding="utf-8",
    )
    calls = []

    def fake_fetch(symbol, timeframe, start, end):
        calls.append((symbol, timeframe, pd.Timestamp(start), pd.Timestamp(end)))
        if timeframe == "D1":
            opens = pd.date_range(
                pd.Timestamp(start).floor("D") + pd.Timedelta(hours=1),
                end, freq="24h", inclusive="left",
            )
            opens = opens[opens >= pd.Timestamp(start)]
        else:
            opens = pd.date_range(start, end, freq=timeframe_duration(timeframe), inclusive="left")
        frame = _bars(list(opens))
        frame.attrs["mt5_provenance"] = {
            "terminal_build": 1234,
            "terminal_version": [5, 0, 1234],
            "metatrader5_package_version": "5.0-test",
            "retrieved_at_utc": "2026-01-06T00:00:00+00:00",
        }
        return frame

    monkeypatch.setattr(run_mt5_bundle, "fetch_market_bars", fake_fetch)
    monkeypatch.setattr(run_mt5_bundle, "_git_version", lambda: "test-commit")
    run_dir = tmp_path / "new-run"
    run_mt5_bundle.main([
        "--symbol", "BTCUSD.test", "--analysis-start", ANALYSIS_START,
        "--analysis-end", ANALYSIS_END, "--cutoff", ANALYSIS_END,
        "--feature-contract", str(contract), "--run-dir", str(run_dir),
    ])
    assert [item[1] for item in calls] == ["H1", "H4", "D1"]
    start = pd.Timestamp(ANALYSIS_START)
    end = pd.Timestamp(ANALYSIS_END)
    for (_, timeframe, actual_start, actual_end), lookback in zip(calls, [2, 1, 1]):
        assert actual_start == start - (lookback + 2) * timeframe_duration(timeframe)
        assert actual_end == end
        source = json.loads((run_dir / timeframe / "manifest.json").read_text(encoding="utf-8"))
        assert source["provenance"]["terminal_build"] == 1234
        assert source["provenance"]["request_end_utc"] == end.isoformat()
        assert source["provenance"]["feature_contract_sha256"] == sha256_file(contract)
    assert load_market_data_bundle(run_dir / "bundle_manifest.json").manifest["symbol"] == "BTCUSD.test"


def test_bundle_cli_rejects_run_dir_outside_local(tmp_path: Path, monkeypatch) -> None:
    from experiments.ep002_market_data import run_mt5_bundle

    contract = tmp_path / "contract.toml"
    contract.write_text(
        "[timeframes.H1]\nmax_lookback_bars = 2\n"
        "[timeframes.H4]\nmax_lookback_bars = 2\n"
        "[timeframes.D1]\nmax_lookback_bars = 2\n",
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(run_mt5_bundle, "fetch_market_bars", lambda *a: called.append(a))
    outside = Path(__file__).resolve().parents[2] / "outside-broker-output"
    assert not outside.exists()
    with pytest.raises(ValueError, match=".local"):
        run_mt5_bundle.main([
            "--symbol", "BTCUSD.test", "--analysis-start", ANALYSIS_START,
            "--analysis-end", ANALYSIS_END, "--cutoff", ANALYSIS_END,
            "--feature-contract", str(contract), "--run-dir", str(outside),
        ])
    assert called == []
    assert not outside.exists()
