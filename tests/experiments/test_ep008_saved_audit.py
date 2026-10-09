from pathlib import Path
import runpy


def test_saved_run_audit_recomputes_frozen_metrics_without_fitting():
    repo = Path(__file__).resolve().parents[2]
    audit = runpy.run_path(
        str(repo / "experiments/ep008_joint_five_observation_predictability/audit_saved.py")
    )["audit"]

    report = audit()

    assert report["source"]["archive_sha256"] == (
        "e39ffa4c4e8cf3207f2b2589ce5ed5aea56253b35e33c66928d1f7ed76a9d941"
    )
    assert report["source"]["observations"] == 6654
    assert report["source"]["duplicate_dates"] == 0
    assert report["source"]["weekdays_without_rate"] == 129
    assert report["source"]["forward_fill_applied"] is False
    assert report["saved_run"]["cold_rows"] == 186
    scores = report["saved_run"]["metrics_recomputed_from_saved_predictions"]
    assert scores["return_mae_bps"]["zero"] == 64.78548406322781
    assert scores["movement_qlike"]["linear"] == 203184.80402315964
    assert scores["movement_linear_floor_clipped_count"] == 3
    assert scores["quantile_mean_pinball_bps"]["boosting"] == 16.589947216430474
