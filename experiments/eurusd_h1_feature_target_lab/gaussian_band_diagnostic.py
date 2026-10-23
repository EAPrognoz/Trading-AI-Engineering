#!/usr/bin/env python3
"""No-fit Gaussian single-observed-return band diagnostic from saved predictions."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/'experiments/eurusd_h1_feature_target_selection_v3/results/eurusd_h1_feature_target_v3_20261022T151237Z'
MAX_ORIGIN=10620
CUTOFF_OPEN=pd.Timestamp('2026-07-03T15:00:00Z')
CUTOFF_END=pd.Timestamp('2026-07-03T16:00:00Z')
FOLDS={'F1':('2025-07-01T00:00:00Z','2025-10-01T00:00:00Z'),'F2':('2025-10-01T00:00:00Z','2026-01-01T00:00:00Z'),'F3':('2026-01-01T00:00:00Z','2026-04-01T00:00:00Z'),'F4':('2026-04-01T00:00:00Z','2026-07-04T00:00:00Z')}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def reconstruct(mean_by_h):
 if set(mean_by_h)!=set(range(1,9)):raise ValueError('need exactly target means H=1..8')
 m=np.array([0.]+[float(mean_by_h[h]) for h in range(1,9)],dtype=float)
 if not np.isfinite(m).all():raise ValueError('nonfinite target means')
 return np.array([h*m[h]-(h-1)*m[h-1] for h in range(1,9)],dtype=float)

def main():
 global BASE
 ap=argparse.ArgumentParser()
 ap.add_argument('run_dir',type=Path)
 ap.add_argument('--v3-run',type=Path,default=BASE,
                 help='matching v3 run containing the nested target manifest')
 ap.add_argument('--artifact-inventory',type=Path,default=HERE/'artifact_inventory.json',
                 help='frozen checksum inventory for the selected v3 run')
 args=ap.parse_args()
 BASE=args.v3_run.resolve()
 out=args.run_dir.resolve()
 if out.parent!=(HERE/'results').resolve():raise RuntimeError('run_dir must be inside this analysis results directory')
 receipt=json.loads((out/'pre_fit_data_receipt.json').read_text())
 if receipt['max_origin_row']!=MAX_ORIGIN or receipt['target_end_max']!='2026-07-03T16:00:00+00:00':raise RuntimeError('pre-fit source cutoff receipt mismatch')
 target_path=BASE/'target_manifest.csv.gz'
 expected=json.loads(args.artifact_inventory.resolve().read_text())['artifact_hashes_sha256']['target_manifest.csv.gz']
 if sha(target_path)!=expected:raise RuntimeError('target manifest changed after pre-fit protocol')
 pred_path=out/'outer_predictions_mean_variance.csv'
 if not pred_path.exists():raise RuntimeError('primary checkpoint-derived predictions are missing')
 pred=pd.read_csv(pred_path)
 need={'outer_fold','origin_row','decision_timestamp','target_timestamp_H8','y_mean_prediction_H8','y_mean_actual_H8','y_var_prediction_H8_raw','y_var_actual_H8_ddof0'}
 if not need.issubset(pred.columns):raise RuntimeError('prediction schema missing fields for Gaussian diagnostic')
 if len(pred)!=6163 or pred.origin_row.max()>MAX_ORIGIN:raise RuntimeError('unexpected validation row count or origin')
 pred.origin_row=pred.origin_row.astype(int)
 if pd.to_datetime(pred.decision_timestamp,utc=True).max()>CUTOFF_END:raise RuntimeError('predictions extend after data cutoff')
 wanted=set(pred.origin_row.tolist())
 nested={}
 with gzip.open(target_path,'rt',newline='',encoding='utf-8') as f:
  rdr=csv.DictReader(f)
  expected_cols={'origin_row','origin_bar_open_timestamp','decision_timestamp','target_timestamp','H','y_mean_log_return_per_observed_transition','y_var_log_return_per_observed_transition_ddof0','target_valid_next_observed_H'}
  if not expected_cols.issubset(rdr.fieldnames):raise RuntimeError('target manifest schema changed')
  for r in rdr:
   o=int(r['origin_row']);h=int(r['H'])
   if o>MAX_ORIGIN:raise RuntimeError('target origin exceeds pre-test cutoff')
   if r['origin_bar_open_timestamp'] and pd.Timestamp(r['origin_bar_open_timestamp'])>CUTOFF_OPEN:raise RuntimeError('target origin open exceeds cutoff')
   if r['decision_timestamp'] and pd.Timestamp(r['decision_timestamp'])>CUTOFF_END:raise RuntimeError('target decision time exceeds cutoff')
   if r['target_timestamp'] and pd.Timestamp(r['target_timestamp'])>CUTOFF_END:raise RuntimeError('target endpoint exceeds cutoff')
   if o in wanted and 1<=h<=8 and r['target_valid_next_observed_H']=='True':
    nested.setdefault(o,{})[h]={'mean':float(r['y_mean_log_return_per_observed_transition']),'variance':float(r['y_var_log_return_per_observed_transition_ddof0']) if r['y_var_log_return_per_observed_transition_ddof0'] else np.nan,'origin_ts':r['origin_bar_open_timestamp'],'decision_ts':r['decision_timestamp'],'target_ts':r['target_timestamp']}
 rows=[];counts=[];checked=0
 for row in pred.itertuples(index=False):
  o=int(row.origin_row);hmap=nested.get(o,{})
  if set(hmap)!=set(range(1,9)):raise RuntimeError(f'missing one or more H1..H8 targets for validation origin {o}')
  if any(not np.isfinite(hmap[h]['mean']) for h in range(1,9)):raise RuntimeError(f'nonfinite nested target at origin {o}')
  ends=[pd.Timestamp(hmap[h]['target_ts']) for h in range(1,9)]
  if any(ends[i]>=ends[i+1] for i in range(7)) or ends[-1]>CUTOFF_END:raise RuntimeError(f'target timestamp order/cutoff failure at origin {o}')
  if any(hmap[h]['decision_ts']!=row.decision_timestamp for h in range(1,9)):raise RuntimeError(f'decision timestamp mismatch at origin {o}')
  if ends[-1]!=pd.Timestamp(row.target_timestamp_H8):raise RuntimeError(f'H8 endpoint mismatch at origin {o}')
  fold_start,fold_end=FOLDS[row.outer_fold]
  if not(pd.Timestamp(fold_start)<=pd.Timestamp(row.decision_timestamp)<pd.Timestamp(fold_end) and ends[-1]<pd.Timestamp(fold_end)):
   raise RuntimeError(f'outer fold boundary failure at origin {o}')
  returns=reconstruct({h:hmap[h]['mean'] for h in range(1,9)})
  actual_mean=float(hmap[8]['mean']);actual_var=float(hmap[8]['variance'])
  if not np.isclose(returns.mean(),actual_mean,rtol=1e-10,atol=1e-14):raise RuntimeError(f'reconstructed H8 mean mismatch at origin {o}')
  if not np.isclose(np.var(returns,ddof=0),actual_var,rtol=1e-10,atol=1e-14):raise RuntimeError(f'reconstructed H8 variance mismatch at origin {o}')
  if not np.isclose(actual_mean,float(row.y_mean_actual_H8),rtol=1e-10,atol=1e-14) or not np.isclose(actual_var,float(row.y_var_actual_H8_ddof0),rtol=1e-10,atol=1e-14):
   raise RuntimeError(f'reconstructed target disagrees with saved primary prediction row at origin {o}')
  checked+=1
  raw=float(row.y_var_prediction_H8_raw);mu=float(row.y_mean_prediction_H8)
  band1=band2=np.nan;inside1=inside2=None
  if raw>0:
   sigma=float(np.sqrt(raw));inside1=(returns>=mu-sigma)&(returns<=mu+sigma);inside2=(returns>=mu-2*sigma)&(returns<=mu+2*sigma)
   band1=float(inside1.mean());band2=float(inside2.mean())
  rows.append({'outer_fold':row.outer_fold,'origin_row':o,'decision_timestamp':row.decision_timestamp,'target_timestamp_H8':row.target_timestamp_H8,'raw_variance_status':'negative' if raw<0 else ('exact_zero' if raw==0 else 'positive'),'mu_predicted_per_observed_transition':mu,'raw_variance_prediction':raw,'sigma_if_positive':float(np.sqrt(raw)) if raw>0 else np.nan,'returns_inside_mu_pm_sigma_fraction':band1,'returns_inside_mu_pm_2sigma_fraction':band2,'covered_count_pm_sigma':int(inside1.sum()) if raw>0 else np.nan,'covered_count_pm_2sigma':int(inside2.sum()) if raw>0 else np.nan})
 for fold in list(FOLDS)+['ALL']:
  cohort=rows if fold=='ALL' else [r for r in rows if r['outer_fold']==fold]
  positive=[r for r in cohort if r['raw_variance_status']=='positive']
  nonpos=[r for r in cohort if r['raw_variance_status']!='positive']
  nreturns=8*len(positive)
  c1=sum(r['covered_count_pm_sigma'] for r in positive);c2=sum(r['covered_count_pm_2sigma'] for r in positive)
  counts.append({'fold':fold,'origins_total':len(cohort),'positive_variance_origins':len(positive),'raw_negative_origins':sum(r['raw_variance_status']=='negative' for r in cohort),'raw_exact_zero_origins':sum(r['raw_variance_status']=='exact_zero' for r in cohort),'raw_nonpositive_origins':len(nonpos),'positive_origin_coverage_of_all':len(positive)/len(cohort) if cohort else 0,'returns_checked_per_band':nreturns,'inside_pm_sigma_returns':int(c1),'micro_coverage_pm_sigma':c1/nreturns if nreturns else None,'mean_origin_coverage_pm_sigma':float(np.mean([r['returns_inside_mu_pm_sigma_fraction'] for r in positive])) if positive else None,'inside_pm_2sigma_returns':int(c2),'micro_coverage_pm_2sigma':c2/nreturns if nreturns else None,'mean_origin_coverage_pm_2sigma':float(np.mean([r['returns_inside_mu_pm_2sigma_fraction'] for r in positive])) if positive else None})
 pd.DataFrame(rows).to_csv(out/'gaussian_band_origin_diagnostics.csv',index=False,float_format='%.17g')
 pd.DataFrame(counts).to_csv(out/'gaussian_band_diagnostic.csv',index=False,float_format='%.17g')
 report=['# Gaussian single-return band diagnostic','',f'Run `{out.name}`; reconstructed and verified H1..H8 nested targets for {checked} saved validation origins from the allowed pre-test manifest. No fit was run.','', 'Working reference: for an individual observed-transition return within the next-eight-observed-return window, compare to `N(saved H8 mean prediction per observed transition, saved raw H8 within-window variance)`. This is not a distribution for the H8 mean or terminal return.','', 'Bands are inclusive. Coverage is calculated only for positive raw predicted variance; negative and exact-zero predictions are reported separately without epsilon. Equal-origin and pooled-return coverage are both shown. Nominal 68.27% and 95.45% are Gaussian reference values, not guaranteed or claimed results. Marginal normality does not imply independence; no `sqrt(H)` or terminal-variance inference is made.','',f'Target reconstruction checks passed: nested H1..H8 timestamps ordered; each H8 endpoint matches the primary prediction row and is before its fold end; reconstructed H8 mean and ddof=0 variance match saved targets at rtol 1e-10 / atol 1e-14.','', '| Fold | Origins | Positive variance | raw<0 / raw=0 | returns / band | ±σ micro / origin | ±2σ micro / origin |','|---|---:|---:|---:|---:|---:|---:|']
 for r in counts:
  a='—' if r['micro_coverage_pm_sigma'] is None else f"{100*r['micro_coverage_pm_sigma']:.2f}% / {100*r['mean_origin_coverage_pm_sigma']:.2f}%"
  b='—' if r['micro_coverage_pm_2sigma'] is None else f"{100*r['micro_coverage_pm_2sigma']:.2f}% / {100*r['mean_origin_coverage_pm_2sigma']:.2f}%"
  report.append(f"| {r['fold']} | {r['origins_total']} | {r['positive_variance_origins']} | {r['raw_negative_origins']} / {r['raw_exact_zero_origins']} | {r['returns_checked_per_band']} | {a} | {b} |")
 (out/'GAUSSIAN_DIAGNOSTIC_RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 summary_path=out/'run_summary.json'
 if summary_path.exists():
  summary=json.loads(summary_path.read_text());summary['status']='complete_primary_and_gaussian_diagnostic';summary['gaussian_target_origins_verified']=checked;summary['gaussian_diagnostic_csv']='gaussian_band_diagnostic.csv';summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'status':'gaussian_diagnostic_complete','origins_verified':checked,'bands':counts},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
