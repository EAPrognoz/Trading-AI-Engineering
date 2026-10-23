"""Synthetic, no-fit checks for summary/target helpers; never reads market artifacts."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_joint_utility as j


def fixture_frame():
    return pd.DataFrame({
        'outer_fold':['F1']*6,
        'origin_row':[10,11,12,13,14,15],
        'actual_sign_H8':[-1,1,0,1,-1,1],
        'pred_sign_H8':[-1,1,1,0,-1,-1],
        'train_majority_H8_sign':[1]*6,
        'y_mean_prediction_H8':[-.001,.002,.003,0,-.002,.001],
        'y_var_prediction_H8_raw':[.1,.2,-.1,0,.3,.4],
        'y_var_prediction_H8_clipped':[.1,.2,0,0,.3,.4],
        'predicted_sigma_H8_positive_raw':[np.sqrt(.1),np.sqrt(.2),np.nan,np.nan,np.sqrt(.3),np.sqrt(.4)],
        'y_var_actual_H8_ddof0':[.12,.19,.21,.15,.4,.31],
        'train_mean_variance_baseline':[.2]*6,
        'actual_abs_cumulative_H8_logreturn_bp':[1,2,3,4,5,6],
        'gross_directional_outcome_bp':[1,2,np.nan,np.nan,5,-6],
        'mean_strength_bin':['weak','medium','strong','weak','medium','strong'],
        'sigma_bin_positive_raw_only':['small','medium','','','large','large'],
        'ratio_bin_positive_raw_only':['low','medium','','','high','high'],
    })


class NoFitHelpersTest(unittest.TestCase):
    def test_nested_means_reconstruct_returns(self):
        r=np.array([.001,-.002,.004,.003,-.001,.002,-.003,.005])
        m={h:float(np.mean(r[:h])) for h in range(1,9)}
        got=j.reconstruct_returns_from_means(m)
        np.testing.assert_allclose(got,r,rtol=0,atol=2e-18)
        self.assertAlmostEqual(float(got.mean()),float(np.mean(r)),places=16)
        self.assertAlmostEqual(float(got.var(ddof=0)),float(np.var(r,ddof=0)),places=16)

    def test_reconstruct_rejects_incomplete_horizons(self):
        with self.assertRaises(ValueError):
            j.reconstruct_returns_from_means({h:0.0 for h in range(1,8)})

    def test_build_metric_tables_all_summary_paths(self):
        frame=fixture_frame()
        tables=j.build_metric_tables(frame,['F1'])
        self.assertEqual(len(tables['grid']),18)
        self.assertEqual(len(tables['nonpositive_rows']),24)
        self.assertEqual(len(tables['ratio_rows']),6)
        self.assertEqual(len(tables['overall_rows']),10)
        agg={r['cell']:r for r in tables['overall_rows'] if r['fold']=='ALL'}
        self.assertEqual(agg['all']['n'],6)
        self.assertEqual(agg['positive_raw_variance']['n'],4)
        self.assertEqual(agg['raw_nonpositive']['n'],2)
        self.assertEqual(agg['raw_negative']['n'],1)
        self.assertEqual(agg['raw_exact_zero']['n'],1)
        self.assertEqual(sum(r['n'] for r in tables['grid'] if r['fold']=='ALL'),4)
        nonpos=next(r for r in tables['nonpositive_rows'] if r['fold']=='ALL' and r['cell']=='all_mean_bins / raw_nonpositive')
        self.assertEqual(nonpos['n'],2)

    def test_empty_summary_path_and_repeated_edges(self):
        frame=fixture_frame()
        row=j.summarize(frame.iloc[0:0], 'F1','test','empty',6,4)
        self.assertEqual(row['n'],0)
        self.assertIsNone(row['h8_hit_rate'])
        self.assertEqual(j.bin_id(1.0,np.array([1.0,1.0])),0)
        self.assertEqual(j.bin_id(1.1,np.array([1.0,1.0])),2)

    def test_no_model_fit_is_invoked(self):
        frame=fixture_frame()
        with mock.patch.object(Ridge,'fit',side_effect=AssertionError('fit must not run in fixture tests')):
            j.build_metric_tables(frame,['F1'])


if __name__=='__main__':
    unittest.main(verbosity=2)
