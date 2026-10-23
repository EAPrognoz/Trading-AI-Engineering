#!/usr/bin/env python3
"""Four frozen H8 variance-only Ridge fits plus fixed mean/variance strata."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,platform,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['NUMEXPR_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/'experiments/eurusd_h1_feature_target_selection_v3/results/eurusd_h1_feature_target_v3_20261022T151237Z'
MEAN=REPO/'experiments/eurusd_h1_horizon_agreement/results/eurusd_h1_horizon_agreement_v1_20261022T170250Z'
PROTOCOL=HERE/'protocol_frozen_pre_fit.json'
INVENTORY_PATH=HERE/'artifact_inventory.json'
FOLDS={'F1':('2025-07-01T00:00:00Z','2025-10-01T00:00:00Z'),'F2':('2025-10-01T00:00:00Z','2026-01-01T00:00:00Z'),'F3':('2026-01-01T00:00:00Z','2026-04-01T00:00:00Z'),'F4':('2026-04-01T00:00:00Z','2026-07-04T00:00:00Z')}
M_LABELS=['weak','medium','strong']; S_LABELS=['small','medium','large']; R_LABELS=['low','medium','high']
CORE={'own_ohlc_log_ratios','same_bar_cross_ratios','observed_transition_gap'}
MAX_ORIGIN=10620
MAX_DECISION_TS='2026-07-03T16:00:00Z'
MAX_TARGET_END='2026-07-03T16:00:00Z'
EXPECTED_PREFIX_SHA256='141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270'


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

def read_gz(p):
 with gzip.open(p,'rt',newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def sign(v): return 1 if v>0 else (-1 if v<0 else 0)

def edges(x):
 x=np.asarray(x,dtype=float)
 if len(x)==0:return np.array([np.nan,np.nan])
 return np.quantile(x,[1/3,2/3],method='linear')

# Alias used below when constructing fold-specific train-only tercile cutoffs.
qedges=edges

def bin_id(x,e):
 if not np.isfinite(x) or not np.isfinite(e).all():return None
 return int(np.searchsorted(e,x,side='left'))

def hit_stats(y,p):
 y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=np.int8)
 use=(y!=0)&(p!=0); n=int(use.sum()); hits=int((y[use]==p[use]).sum()) if n else 0
 acc=hits/n if n else None; ba=None
 if n and set(y[use].tolist())=={-1,1}:
  ya=y[use]; pa=p[use]; ba=float(np.mean([np.mean(pa[ya==-1]==-1),np.mean(pa[ya==1]==1)]))
 return hits,acc,n,ba

def safe_mean(a):
 a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
 return float(a.mean()) if len(a) else None

def safe_median(a):
 a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
 return float(np.median(a)) if len(a) else None

def safe_q(a,q):
 a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
 return float(np.quantile(a,q,method='linear')) if len(a) else None

def write_csv(p,rows):
 if not rows:p.write_text('',encoding='utf-8');return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def atomic_npz(path,arrays):
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('wb') as f:
  np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 os.replace(tmp,path)

def fmt(x,spec='.6g'):
 if x is None or not np.isfinite(float(x)): return '—'
 return format(float(x),spec)

def summarize(frame,fold,group,cell,full_n,positive_n):
 n=len(frame)
 if n:
  y=frame.actual_sign_H8.to_numpy(dtype=np.int8); p=frame.pred_sign_H8.to_numpy(dtype=np.int8)
  hits,acc,evaln,ba=hit_stats(y,p)
  maj=frame.train_majority_H8_sign.to_numpy(dtype=np.int8)
  mhits,macc,maj_n,maj_ba=hit_stats(y,maj)
  actual_var=frame.y_var_actual_H8_ddof0.to_numpy(dtype=float)
  pred_var=frame.y_var_prediction_H8_clipped.to_numpy(dtype=float)
  base_var=frame.train_mean_variance_baseline.to_numpy(dtype=float)
  var_mae=float(np.mean(np.abs(pred_var-actual_var)))
  base_mae=float(np.mean(np.abs(base_var-actual_var)))
  var_skill=1-var_mae/base_mae if base_mae>0 else None
  actual_var_mean=float(np.mean(actual_var)); pred_var_mean=float(np.mean(pred_var))
  calibration=pred_var_mean/actual_var_mean if actual_var_mean>0 else None
  actual_abs=frame.actual_abs_cumulative_H8_logreturn_bp.to_numpy(dtype=float)
  gross=frame.gross_directional_outcome_bp.to_numpy(dtype=float)
  gross=gross[np.isfinite(gross)]
  absmean=np.abs(frame.y_mean_prediction_H8.to_numpy(dtype=float))
  absmean_cum=absmean*8*10000
  sigma=frame.predicted_sigma_H8_positive_raw.to_numpy(dtype=float)
  zero_actual=int((y==0).sum());zero_pred=int((p==0).sum())
  zero_var=int((frame.y_var_prediction_H8_raw.to_numpy(dtype=float)==0).sum())
  neg_var=int((frame.y_var_prediction_H8_raw.to_numpy(dtype=float)<0).sum())
  zero_actual_var=int((actual_var==0).sum())
  buy=int((p==1).sum());sell=int((p==-1).sum())
 else:
  y=np.array([],dtype=np.int8);p=y;hits=evaln=mhits=maj_n=zero_actual=zero_pred=zero_var=neg_var=zero_actual_var=buy=sell=0
  acc=ba=macc=maj_ba=var_skill=calibration=var_mae=base_mae=actual_var_mean=pred_var_mean=None
  actual_abs=np.array([]);actual_var=np.array([]);gross=np.array([]);absmean=np.array([]);absmean_cum=np.array([]);sigma=np.array([])
 return {
  'fold':fold,'group':group,'cell':cell,'n':n,'full_validation_origins':full_n,
  'coverage_full_validation':n/full_n if full_n else 0,
  'positive_raw_variance_origins':positive_n,'share_positive_variance_subset':n/positive_n if positive_n and group!='raw_nonpositive' else None,
  'h8_zero_actual_mean_count':zero_actual,'h8_zero_mean_forecast_count':zero_pred,
  'raw_variance_exact_zero_prediction_count':zero_var,'raw_variance_negative_prediction_count':neg_var,'realized_variance_zero_count':zero_actual_var,
  'h8_buy_count':buy,'h8_sell_count':sell,'h8_hits':hits,'h8_direction_eval_n':evaln,'h8_hit_rate':acc,'h8_balanced_accuracy':ba,
  'train_majority_hits':mhits,'train_majority_eval_n':maj_n,'train_majority_accuracy':macc,'train_majority_balanced_accuracy':maj_ba,
  'predicted_abs_mean_per_return_mean':safe_mean(absmean),'predicted_abs_mean_per_return_median':safe_median(absmean),
  'predicted_abs_mean_cumulative_bp_mean':safe_mean(absmean_cum),'predicted_abs_mean_cumulative_bp_median':safe_median(absmean_cum),
  'predicted_sigma_mean':safe_mean(sigma),'predicted_sigma_median':safe_median(sigma),
  'realized_abs_H8_cumulative_bp_mean':safe_mean(actual_abs),'realized_abs_H8_cumulative_bp_median':safe_median(actual_abs),'realized_abs_H8_cumulative_bp_p90':safe_q(actual_abs,.9),
  'realized_variance_mean':actual_var_mean,'realized_variance_median':safe_median(actual_var),
  'clipped_predicted_variance_mean':pred_var_mean,'variance_mae':var_mae,'train_mean_variance_baseline_mae':base_mae,'variance_mae_skill':var_skill,'variance_mean_calibration_ratio':calibration,
  'gross_directional_outcome_n':len(gross),'gross_directional_outcome_bp_median':safe_median(gross),'gross_directional_outcome_bp_p10':safe_q(gross,.1),
 }

def build_metric_tables(df,fold_names):
 positive=df[df.y_var_prediction_H8_raw>0].copy()
 nonpos=df[df.y_var_prediction_H8_raw<=0].copy()
 fulln={f:int((df.outer_fold==f).sum()) for f in fold_names};fulln['ALL']=len(df)
 posn={f:int((positive.outer_fold==f).sum()) for f in fold_names};posn['ALL']=len(positive)
 grid=[];nonposrows=[];ratiorows=[];overallrows=[]
 for f in list(fold_names)+['ALL']:
  base=positive if f=='ALL' else positive[positive.outer_fold==f]
  for ml in M_LABELS:
   for sl in S_LABELS:
    c=base[(base.mean_strength_bin==ml)&(base.sigma_bin_positive_raw_only==sl)]
    grid.append(summarize(c,f,'positive_variance_3x3',ml+' x '+sl,fulln[f],posn[f]))
  nb=nonpos if f=='ALL' else nonpos[nonpos.outer_fold==f]
  for ml in M_LABELS+['all_mean_bins']:
   cohort=nb if ml=='all_mean_bins' else nb[nb.mean_strength_bin==ml]
   for status,c in [('negative',cohort[cohort.y_var_prediction_H8_raw<0]),('exact_zero',cohort[cohort.y_var_prediction_H8_raw==0]),('raw_nonpositive',cohort)]:
    nonposrows.append(summarize(c,f,'raw_nonpositive',ml+' / '+status,fulln[f],posn[f]))
  for label in R_LABELS:
   ratiorows.append(summarize(base[base.ratio_bin_positive_raw_only==label],f,'positive_variance_ratio',label,fulln[f],posn[f]))
  allf=df if f=='ALL' else df[df.outer_fold==f]
  posf=positive if f=='ALL' else positive[positive.outer_fold==f]
  nonposf=nonpos if f=='ALL' else nonpos[nonpos.outer_fold==f]
  for label,c in [('all',allf),('positive_raw_variance',posf),('raw_nonpositive',nonposf),('raw_negative',nonposf[nonposf.y_var_prediction_H8_raw<0]),('raw_exact_zero',nonposf[nonposf.y_var_prediction_H8_raw==0])]:
   overallrows.append(summarize(c,f,'variance_overall',label,fulln[f],posn[f]))
 return {'positive':positive,'nonpositive':nonpos,'fulln':fulln,'positive_n':posn,'grid':grid,'nonpositive_rows':nonposrows,'ratio_rows':ratiorows,'overall_rows':overallrows}

def reconstruct_returns_from_means(mean_by_h):
 """Recover one observed-transition return per horizon from nested mean targets."""
 if set(mean_by_h)!=set(range(1,9)):raise ValueError('Need all H=1..8 mean targets.')
 m=[0.0]+[float(mean_by_h[h]) for h in range(1,9)]
 if not np.isfinite(m).all():raise ValueError('Nonfinite nested mean target.')
 return np.asarray([h*m[h]-(h-1)*m[h-1] for h in range(1,9)],dtype=float)

def load_checkpoint(path):
 with np.load(path,allow_pickle=False) as z:
  q={k:z[k].copy() for k in z.files}
 if 'mean_edges' not in q:q['mean_edges']=qedges(np.abs(q['mean_prediction_train']))
 if 'sigma_edges' not in q:
  positive=q['raw_variance_train']>0
  q['sigma_edges']=qedges(np.sqrt(q['raw_variance_train'][positive]))
 if 'ratio_edges' not in q:
  positive=q['raw_variance_train']>0
  ratio=np.abs(q['mean_prediction_train'][positive])/np.sqrt(q['raw_variance_train'][positive])
  q['ratio_edges']=qedges(ratio)
 if not np.isfinite(q['mean_edges']).all() or not np.isfinite(q['sigma_edges']).all() or not np.isfinite(q['ratio_edges']).all():
  raise RuntimeError(f'Checkpoint has insufficient train-only predictions for frozen terciles: {path.name}')
 if any(k not in q for k in ['train_origins','validation_origins','coef','intercept','raw_variance_train','raw_variance_validation','mean_actual_validation','variance_actual_validation']):
  raise RuntimeError(f'Checkpoint schema incomplete: {path.name}')
 atomic_npz(path,q)
 return {'train':q['train_origins'],'valid':q['validation_origins'],'fitid':int(q['fit_id'][0]),'xm':q['x_mean'],'xs':q['x_scale'],'coef':q['coef'],'intercept':float(q['intercept'][0]),'y_mean':q['y_mean'],'y_scale':q['y_scale'],'raw_train':q['raw_variance_train'],'raw_valid':q['raw_variance_validation'],'mean_train':q['mean_prediction_train'],'mean_valid':q['mean_prediction_validation'],'mean_actual_valid':q['mean_actual_validation'],'variance_actual_valid':q['variance_actual_validation'],'origin_ts_valid':q['origin_timestamp_validation'],'decision_ts_valid':q['decision_timestamp_validation'],'target_ts_valid':q['target_timestamp_validation'],'gap_count_valid':q['gap_count_validation'],'gap_cross_valid':q['gap_cross_validation'],'mean_edges':q['mean_edges'],'sigma_edges':q['sigma_edges'],'ratio_edges':q['ratio_edges'],'train_mean_var':float(q['train_mean_variance_baseline'][0]),'majority':int(q['majority_sign'][0]),'train_hash':str(q['train_origins_sha256'][0]),'val_hash':str(q['validation_origins_sha256'][0])}

def main():
 global BASE,MEAN,PROTOCOL,INVENTORY_PATH
 wall0=time.perf_counter();cpu0=time.process_time()
 parser=argparse.ArgumentParser()
 parser.add_argument('--base-run',type=Path,default=BASE,help='pinned v3 feature/target/split run directory')
 parser.add_argument('--mean-run',type=Path,default=MEAN,help='pinned H4/H8/H24 agreement run directory')
 parser.add_argument('--protocol',type=Path,default=PROTOCOL,help='frozen joint-utility protocol')
 parser.add_argument('--artifact-inventory',type=Path,default=INVENTORY_PATH,help='frozen checksum inventory for upstream files')
 parser.add_argument('--output-root',type=Path,default=HERE/'results')
 parser.add_argument('--assemble-only',type=Path,default=None,help='assemble metrics from a run directory with saved per-fold checkpoints; performs no fit')
 parser.add_argument('--preflight-only',action='store_true',help='validate allowed-prefix sources/schemas/boundaries, then exit without a fit')
 args=parser.parse_args()
 BASE=args.base_run.resolve(); MEAN=args.mean_run.resolve(); PROTOCOL=args.protocol.resolve(); INVENTORY_PATH=args.artifact_inventory.resolve()
 output_root=args.output_root.resolve()
 if args.assemble_only and args.preflight_only:raise RuntimeError('Choose either assemble-only or preflight-only.')
 protocol=json.loads(PROTOCOL.read_text())
 if protocol.get('status')!='protocol_frozen_after_review_before_fit':raise RuntimeError('Protocol is not frozen.')
 inventory=json.loads(INVENTORY_PATH.read_text())
 paths={
  'source_prefix_run_summary.json':BASE/'run_summary.json','feature_matrices.npz':BASE/'feature_matrices.npz','feature_catalog.json':BASE/'feature_catalog.json',
  'split_membership.csv.gz':BASE/'split_membership.csv.gz','target_manifest.csv.gz':BASE/'target_manifest.csv.gz',
  'consensus_outer_predictions.csv':MEAN/'outer_predictions.csv','consensus_fit_index.csv':MEAN/'fit_index.csv',
  'consensus_model_coefficients_and_scalers.npz':MEAN/'model_coefficients_and_scalers.npz','consensus_run_summary.json':MEAN/'run_summary.json'}
 for name,expected in inventory['artifact_hashes_sha256'].items():
  if sha(paths[name])!=expected:raise RuntimeError('Input changed after protocol lock: '+name)
 source_run=json.loads(paths['source_prefix_run_summary.json'].read_text());mean_run=json.loads(paths['consensus_run_summary.json'].read_text())
 if source_run['test_rows_read'] or mean_run['test_rows_read']:raise RuntimeError('Unexpected test access metadata.')
 if source_run['read_rows']!=MAX_ORIGIN+1 or source_run['prefix_sha256_verified']!=EXPECTED_PREFIX_SHA256:
  raise RuntimeError('Source prefix row count/hash differs from the frozen pre-test cutoff.')
 catalog=json.loads(paths['feature_catalog.json'].read_text())
 ix=np.asarray([i for i,x in enumerate(catalog['schema']['16']) if x['group'] in CORE],dtype=int)
 if len(ix)!=21:raise RuntimeError('Core input count changed.')
 with np.load(paths['feature_matrices.npz'],allow_pickle=False) as z:
  Xall=np.asarray(z['X_W16'],dtype=float);valid=np.asarray(z['valid_W16'],dtype=bool)
 if Xall.shape!=(MAX_ORIGIN+1,177) or valid.shape!=(MAX_ORIGIN+1,):raise RuntimeError('Feature matrix is not the exact frozen pre-test prefix.')
 X=Xall[:,ix]
 split=[r for r in read_gz(paths['split_membership.csv.gz']) if r['split_id']=='outer']
 target={};manifest_receipt={'rows':0,'origin_min':None,'origin_max':None,'decision_min':None,'decision_max':None,'nonempty_target_end_max':None,'valid_h8_count':0}
 with gzip.open(paths['target_manifest.csv.gz'],'rt',newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   manifest_receipt['rows']+=1
   o_all=int(r['origin_row']); origin_ts=r['origin_bar_open_timestamp']; decision_ts=r['decision_timestamp']
   if o_all>MAX_ORIGIN or pd.Timestamp(origin_ts)>pd.Timestamp('2026-07-03T15:00:00Z') or pd.Timestamp(decision_ts)>pd.Timestamp(MAX_DECISION_TS):
    raise RuntimeError('Target manifest contains an origin/decision after the frozen prefix cutoff.')
   if manifest_receipt['origin_min'] is None:manifest_receipt['origin_min']=origin_ts
   manifest_receipt['origin_max']=origin_ts
   if manifest_receipt['decision_min'] is None:manifest_receipt['decision_min']=decision_ts
   manifest_receipt['decision_max']=decision_ts
   target_end=r['target_timestamp']
   if target_end:
    if pd.Timestamp(target_end)>pd.Timestamp(MAX_TARGET_END):raise RuntimeError('Target endpoint exceeds the frozen pre-test cutoff.')
    manifest_receipt['nonempty_target_end_max']=max(manifest_receipt['nonempty_target_end_max'] or target_end,target_end)
   if int(r['H'])==8 and r['target_valid_next_observed_H']=='True':
    o=int(r['origin_row']); manifest_receipt['valid_h8_count']+=1
    target[o]={'mean':float(r['y_mean_log_return_per_observed_transition']),'var':float(r['y_var_log_return_per_observed_transition_ddof0']),'origin_ts':origin_ts,'decision_ts':decision_ts,'target_ts':target_end,'gap_count':int(r['target_non_one_hour_transition_count']),'gap_cross':r['target_crosses_non_one_hour_transition']=='True'}
    if not np.isfinite(target[o]['mean']) or not np.isfinite(target[o]['var']) or target[o]['var']<0:raise RuntimeError('Nonfinite or invalid H8 target.')
 if manifest_receipt['origin_min']!='2024-10-16T00:00:00+00:00' or manifest_receipt['origin_max']!='2026-07-03T15:00:00+00:00' or manifest_receipt['decision_max']!='2026-07-03T16:00:00+00:00' or manifest_receipt['nonempty_target_end_max']!='2026-07-03T16:00:00+00:00':
  raise RuntimeError('Manifest timestamp receipt does not match the frozen cutoff.')
 predictions=pd.read_csv(paths['consensus_outer_predictions.csv']);predictions.origin_row=predictions.origin_row.astype(int)
 fitrows=list(csv.DictReader(paths['consensus_fit_index.csv'].open()))
 meanfit={r['outer_fold']:r for r in fitrows if int(r['H'])==8}
 saved=np.load(paths['consensus_model_coefficients_and_scalers.npz'],allow_pickle=False)
 if len(predictions)!=6163:raise RuntimeError('Outer prediction count mismatch.')
 required_pred={'outer_fold','origin_row','decision_timestamp','y_mean_prediction_H8','y_mean_actual_H8'}
 if not required_pred.issubset(predictions.columns):raise RuntimeError('Saved mean-prediction schema mismatch.')
 if pd.to_datetime(predictions['decision_timestamp'],utc=True).max()>pd.Timestamp(MAX_DECISION_TS):raise RuntimeError('Saved mean predictions exceed the frozen cutoff.')
 if set(predictions['outer_fold'])!=set(FOLDS):raise RuntimeError('Unexpected outer fold set in saved predictions.')

 # Complete all fold, target, scaler and replay checks before the first new fit.
 expected_val={'F1':1547,'F2':1534,'F3':1481,'F4':1601}
 for f,(st,en) in FOLDS.items():
  rows=[r for r in split if r['outer_fold']==f]
  tr=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='train'),dtype=np.int64)
  va=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='validation'),dtype=np.int64)
  if len(va)!=expected_val[f] or any(int(o) not in target for o in np.r_[tr,va]):raise RuntimeError(f'Fold sample/target schema check failed: {f}.')
  for r in rows:
   dec=pd.Timestamp(r['decision_timestamp']);t24=pd.Timestamp(r['target24_timestamp'])
   if dec>pd.Timestamp(MAX_DECISION_TS) or t24>pd.Timestamp(MAX_TARGET_END):raise RuntimeError(f'Fold row exceeds frozen data cutoff: {f}.')
   start=pd.Timestamp(st);end=pd.Timestamp(en)
   if r['partition']=='train' and not(dec<start and t24<start):raise RuntimeError(f'Train boundary/purge violation: {f}.')
   if r['partition']=='validation' and not(start<=dec<end and t24<end):raise RuntimeError(f'Validation boundary/purge violation: {f}.')
  if not valid[np.r_[tr,va]].all() or not np.isfinite(X[np.r_[tr,va]]).all():raise RuntimeError(f'Invalid/nonfinite core feature in fold: {f}.')
  mf=meanfit[f];prefix=f'{int(mf["fit_id"]):02d}_{f}_H8'
  if hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest()!=mf['train_origins_sha256'] or hashlib.sha256(va.astype('<i8').tobytes()).hexdigest()!=mf['validation_origins_sha256']:
   raise RuntimeError(f'Fold origin hash mismatch: {f}.')
  xm=np.asarray(saved[f'{prefix}_x_mean'],dtype=float).ravel();xs=np.asarray(saved[f'{prefix}_x_scale'],dtype=float).ravel();coef=np.asarray(saved[f'{prefix}_coef'],dtype=float).ravel()
  intercept=float(np.asarray(saved[f'{prefix}_intercept']).ravel()[0]);ym=float(np.asarray(saved[f'{prefix}_y_mean']).ravel()[0]);ys=float(np.asarray(saved[f'{prefix}_y_scale']).ravel()[0])
  if xm.shape!=(len(ix),) or xs.shape!=(len(ix),) or coef.shape!=(len(ix),) or not np.isfinite(np.r_[xm,xs,coef,intercept,ym,ys]).all() or np.any(xs<=0) or ys<=0:
   raise RuntimeError(f'Saved mean model/scaler shape or numeric check failed: {f}.')
  replay=(((X[va]-xm)/xs)@coef+intercept)*ys+ym
  observed=predictions[predictions.outer_fold==f].set_index('origin_row').loc[va].y_mean_prediction_H8.to_numpy(float)
  if not np.allclose(replay,observed,rtol=1e-10,atol=1e-14):raise RuntimeError(f'Saved H8 mean prediction replay mismatch: {f}.')
  if np.any([target[int(o)]['var']<0 for o in tr]) or not np.isfinite([target[int(o)]['var'] for o in tr]).all():raise RuntimeError(f'Train variance target numeric check failed: {f}.')

 if args.preflight_only:
  print(json.dumps({'status':'preflight_passed_no_fit','test_rows_after_cutoff_read':0,'prefix_sha256':EXPECTED_PREFIX_SHA256,'source_rows':source_run['read_rows'],'feature_row_indices':[0,MAX_ORIGIN],'feature_origin_min':manifest_receipt['origin_min'],'feature_origin_max':manifest_receipt['origin_max'],'decision_timestamp_min':manifest_receipt['decision_min'],'decision_timestamp_max':manifest_receipt['decision_max'],'target_endpoint_max':manifest_receipt['nonempty_target_end_max'],'target_valid_h8_count':manifest_receipt['valid_h8_count'],'manifest_rows':manifest_receipt['rows'],'outer_validation_counts':expected_val,'fold_boundary_and_h24_purge_checks':'passed for every fold','saved_mean_prediction_replay':'passed for every fold','feature_count':len(ix)},indent=2,ensure_ascii=False))
  return

 # All source/schema/boundary checks complete before any variance fit.
 if args.assemble_only:
  out=args.assemble_only.resolve()
  if out.parent!=output_root or not (out/'checkpoints').is_dir():raise RuntimeError('assemble-only path is not a known run checkpoint directory.')
  runid=out.name;checkpoints=out/'checkpoints'
 else:
  runid='eurusd_h1_mean_variance_joint_utility_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  out=output_root/runid;out.mkdir(parents=True,exist_ok=False)
  checkpoints=out/'checkpoints';checkpoints.mkdir()
  atomic_json(out/'pre_fit_data_receipt.json',{'status':'preflight_passed_before_fit','source_prefix_sha256':EXPECTED_PREFIX_SHA256,'max_origin_row':MAX_ORIGIN,'feature_origin_min':manifest_receipt['origin_min'],'feature_origin_max':manifest_receipt['origin_max'],'decision_max':manifest_receipt['decision_max'],'target_end_max':manifest_receipt['nonempty_target_end_max'],'manifest_rows':manifest_receipt['rows'],'valid_h8_targets':manifest_receipt['valid_h8_count'],'test_rows_after_cutoff_read':0,'features_shape':[int(Xall.shape[0]),int(Xall.shape[1])],'core_feature_count':len(ix),'outer_prediction_rows':len(predictions),'fold_counts':{f:int((predictions.outer_fold==f).sum()) for f in FOLDS}})
  atomic_json(HERE/'latest_run.json',{'run_id':runid,'results_dir':str(out.relative_to(REPO))})

 if not args.assemble_only:
  fold={} ; folds_summary={}
  for f,(st,en) in FOLDS.items():
   rows=[r for r in split if r['outer_fold']==f]
   tr=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='train'),dtype=np.int64)
   va=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='validation'),dtype=np.int64)
   if any(int(o) not in target for o in np.r_[tr,va]):raise RuntimeError(f'Missing H8 target in {f}.')
   if not np.isfinite(X[np.r_[tr,va]]).all():raise RuntimeError(f'Nonfinite core features in {f}.')
   # Verify H24 purge and H8 label availability before fitting.
   start=pd.Timestamp(st);end=pd.Timestamp(en)
   for r in rows:
    dec=pd.Timestamp(r['decision_timestamp']); t24=pd.Timestamp(r['target24_timestamp'])
    if r['partition']=='train' and not(dec<start and t24<start):raise RuntimeError(f'Train target boundary violation {f}.')
    if r['partition']=='validation' and not(start<=dec<end and t24<end):raise RuntimeError(f'Validation target boundary violation {f}.')
   mf=meanfit[f];fitid=int(mf['fit_id']);prefix=f'{fitid:02d}_{f}_H8'
   th=hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest();vh=hashlib.sha256(va.astype('<i8').tobytes()).hexdigest()
   if th!=mf['train_origins_sha256'] or vh!=mf['validation_origins_sha256']:raise RuntimeError(f'Origin hash mismatch {f}.')
   xm=np.asarray(saved[f'{prefix}_x_mean'],dtype=float).ravel();xs=np.asarray(saved[f'{prefix}_x_scale'],dtype=float).ravel()
   mc=np.asarray(saved[f'{prefix}_coef'],dtype=float).ravel();mi=float(np.asarray(saved[f'{prefix}_intercept']).ravel()[0])
   ym=float(np.asarray(saved[f'{prefix}_y_mean']).ravel()[0]);ys=float(np.asarray(saved[f'{prefix}_y_scale']).ravel()[0])
   mean_train=(((X[tr]-xm)/xs)@mc+mi)*ys+ym
   mean_val=(((X[va]-xm)/xs)@mc+mi)*ys+ym
   old=predictions[predictions.outer_fold==f].set_index('origin_row').loc[va]
   if not np.allclose(mean_val,old.y_mean_prediction_H8.to_numpy(float),rtol=1e-10,atol=1e-14):raise RuntimeError(f'Mean model replay mismatch {f}.')
   if not np.allclose([target[int(o)]['mean'] for o in va],old.y_mean_actual_H8.to_numpy(float),rtol=1e-10,atol=1e-14):raise RuntimeError(f'H8 target mismatch mean manifest vs saved prediction {f}.')
   ytrain=np.asarray([target[int(o)]['var'] for o in tr],dtype=float).reshape(-1,1)
   if np.any(ytrain<0) or not np.isfinite(ytrain).all():raise RuntimeError(f'Invalid variance target train {f}.')
   ysc=StandardScaler().fit(ytrain)
   mdl=Ridge(alpha=1.0,fit_intercept=True,solver='cholesky')
   mdl.fit((X[tr]-xm)/xs,ysc.transform(ytrain).ravel())
   rawtr=ysc.inverse_transform(mdl.predict((X[tr]-xm)/xs).reshape(-1,1)).ravel()
   rawva=ysc.inverse_transform(mdl.predict((X[va]-xm)/xs).reshape(-1,1)).ravel()
   if not np.isfinite(rawtr).all() or not np.isfinite(rawva).all():raise RuntimeError(f'Nonfinite variance forecast {f}.')
   valid_targets=[target[int(o)] for o in va]
   cp_core={'train_origins':tr,'validation_origins':va,'x_mean':xm,'x_scale':xs,'coef':np.asarray(mdl.coef_,dtype=float).ravel(),'intercept':np.asarray([float(mdl.intercept_)]),'y_mean':np.asarray(ysc.mean_,dtype=float).ravel(),'y_scale':np.asarray(ysc.scale_,dtype=float).ravel(),'raw_variance_train':rawtr,'raw_variance_validation':rawva,'mean_prediction_train':mean_train,'mean_prediction_validation':mean_val,'mean_actual_validation':np.asarray([t['mean'] for t in valid_targets],dtype=float),'variance_actual_validation':np.asarray([t['var'] for t in valid_targets],dtype=float),'origin_timestamp_validation':np.asarray([t['origin_ts'] for t in valid_targets],dtype='U32'),'decision_timestamp_validation':np.asarray([t['decision_ts'] for t in valid_targets],dtype='U32'),'target_timestamp_validation':np.asarray([t['target_ts'] for t in valid_targets],dtype='U32'),'gap_count_validation':np.asarray([t['gap_count'] for t in valid_targets],dtype=np.int16),'gap_cross_validation':np.asarray([t['gap_cross'] for t in valid_targets],dtype=bool),'train_mean_variance_baseline':np.asarray([float(np.mean(ytrain))]),'majority_sign':np.asarray([int(mf['majority_H8_prediction_sign'])]),'train_origins_sha256':np.asarray([th]),'validation_origins_sha256':np.asarray([vh]),'fit_id':np.asarray([fitid]),'alpha':np.asarray([1.0])}
   atomic_npz(checkpoints/f'{f}_variance_checkpoint.npz',cp_core)
   atomic_json(checkpoints/f'{f}_fit_receipt.json',{'fold':f,'fit_id':fitid,'fit_completed_utc':datetime.now(timezone.utc).isoformat(),'target':'H8_population_variance_ddof0','n_train':len(tr),'n_validation':len(va),'alpha':1.0,'solver':'cholesky','x_scaler':'reused saved mean H8 train-only scaler','y_scaler':'variance H8 train-only StandardScaler','train_origins_sha256':th,'validation_origins_sha256':vh,'checkpoint_saved_immediately_after_fit':True,'cutoff_origin_row':MAX_ORIGIN,'cutoff_target_end':MAX_TARGET_END})
   meanvar=float(np.mean(ytrain))
   sigtr=np.sqrt(rawtr[rawtr>0]);ratioraw=np.abs(mean_train[rawtr>0])/sigtr if (rawtr>0).any() else np.array([])
   me=qedges(np.abs(mean_train));se=qedges(sigtr);re=qedges(ratioraw)
   if not np.isfinite(me).all() or not np.isfinite(se).all() or not np.isfinite(re).all():raise RuntimeError(f'Train positive variance predictions insufficient for terciles in {f}.')
   cp={**cp_core,'mean_edges':me,'sigma_edges':se,'ratio_edges':re}
   atomic_npz(checkpoints/f'{f}_variance_checkpoint.npz',cp)
   fold[f]={'train':tr,'valid':va,'fitid':fitid,'xm':xm,'xs':xs,'ysc':ysc,'model':mdl,'raw_train':rawtr,'raw_valid':rawva,'mean_train':mean_train,'mean_valid':mean_val,'mean_edges':me,'sigma_edges':se,'ratio_edges':re,'train_mean_var':meanvar,'majority':int(mf['majority_H8_prediction_sign']),'train_hash':th,'val_hash':vh}
   folds_summary[f]={'n_train':len(tr),'n_validation':len(va),'mean_positive_train_var_predictions':int((rawtr>0).sum()),'mean_nonpositive_train_var_predictions':int((rawtr<=0).sum()),'mean_negative_train_var_predictions':int((rawtr<0).sum()),'mean_zero_train_var_predictions':int((rawtr==0).sum())}

 # Rehydrate fitted states only from the per-fold checkpoints; downstream metrics use these saved checkpoints.
 fold={}
 for f in FOLDS:
  cp_path=checkpoints/f'{f}_variance_checkpoint.npz'
  if not cp_path.exists():raise RuntimeError(f'Missing checkpoint after fit: {f}')
  fold[f]=load_checkpoint(cp_path)
 params={};fit_records=[];thresholds=[];outrows=[]
 for f in FOLDS:
  d=fold[f];prefix=f'{d["fitid"]:02d}_{f}_H8_variance'
  params[prefix+'_coef']=d['coef'];params[prefix+'_intercept']=np.asarray([d['intercept']])
  params[prefix+'_x_mean']=d['xm'];params[prefix+'_x_scale']=d['xs'];params[prefix+'_y_mean']=d['y_mean'];params[prefix+'_y_scale']=d['y_scale']
  fit_records.append({'fit_id':len(fit_records),'outer_fold':f,'target':'H8_population_variance_ddof0','W':16,'H':8,'mask_id':'core','n_features':len(ix),'n_train':len(d['train']),'n_validation':len(d['valid']),'alpha':1.0,'solver':'cholesky','x_scaler':'reused saved H8 mean train-only scaler','y_scaler':'new train-only variance StandardScaler','train_origins_sha256':d['train_hash'],'validation_origins_sha256':d['val_hash'],'train_mean_variance_baseline':d['train_mean_var'],'raw_train_positive_n':int((d['raw_train']>0).sum()),'raw_train_negative_n':int((d['raw_train']<0).sum()),'raw_train_exact_zero_n':int((d['raw_train']==0).sum())})
  mi=d['mean_edges'];si=d['sigma_edges'];ri=d['ratio_edges']
  bmi=np.searchsorted(mi,np.abs(d['mean_train']),side='left');bsi=np.searchsorted(si,np.sqrt(d['raw_train'][d['raw_train']>0]),side='left');bri=np.searchsorted(ri,(np.abs(d['mean_train'][d['raw_train']>0])/np.sqrt(d['raw_train'][d['raw_train']>0])),side='left')
  thresholds.append({'fold':f,'n_train':len(d['train']),'mean_q33_abs_per_transition':float(mi[0]),'mean_q67_abs_per_transition':float(mi[1]),'mean_q33_abs_cumulative_bp':float(mi[0]*8*10000),'mean_q67_abs_cumulative_bp':float(mi[1]*8*10000),'train_mean_bin_n':[int((bmi==i).sum()) for i in range(3)],'sigma_q33':float(si[0]),'sigma_q67':float(si[1]),'positive_train_sigma_n':int((d['raw_train']>0).sum()),'train_sigma_bin_n':[int((bsi==i).sum()) for i in range(3)],'ratio_q33':float(ri[0]),'ratio_q67':float(ri[1]),'positive_train_ratio_n':int((d['raw_train']>0).sum()),'train_ratio_bin_n':[int((bri==i).sum()) for i in range(3)],'raw_train_negative_n':int((d['raw_train']<0).sum()),'raw_train_zero_n':int((d['raw_train']==0).sum())})
  for j,o in enumerate(d['valid']):
   mu=float(d['mean_valid'][j]);raw=float(d['raw_valid'][j]);vclip=max(0.0,raw);sigma=float(np.sqrt(raw)) if raw>0 else np.nan
   mb=bin_id(abs(mu),mi);sb=bin_id(sigma,si) if raw>0 else None; ratio=abs(mu)/sigma if raw>0 else np.nan; rb=bin_id(ratio,ri) if raw>0 else None
   actual=float(d['mean_actual_valid'][j]);actual_var=float(d['variance_actual_valid'][j]);predsign=sign(mu);gross=8*actual*predsign*10000 if predsign else np.nan
   outrows.append({'outer_fold':f,'origin_row':int(o),'origin_bar_open_timestamp':str(d['origin_ts_valid'][j]),'decision_timestamp':str(d['decision_ts_valid'][j]),'target_timestamp_H8':str(d['target_ts_valid'][j]),
     'y_mean_prediction_H8':mu,'y_mean_actual_H8':actual,'pred_sign_H8':predsign,'actual_sign_H8':sign(actual),
     'y_var_prediction_H8_raw':raw,'y_var_prediction_H8_clipped':vclip,'predicted_sigma_H8_positive_raw':sigma,'y_var_actual_H8_ddof0':actual_var,
     'raw_variance_status':'negative' if raw<0 else ('exact_zero' if raw==0 else 'positive'),
     'mean_strength_bin':M_LABELS[mb],'sigma_bin_positive_raw_only':S_LABELS[sb] if sb is not None else '',
     'absmean_over_sigma_positive_raw_only':ratio,'ratio_bin_positive_raw_only':R_LABELS[rb] if rb is not None else '',
     'H8_mean_actual_zero':actual==0,'H8_mean_prediction_zero':mu==0,'H8_variance_actual_zero':actual_var==0,
     'H8_target_gap_count_audit_only':int(d['gap_count_valid'][j]),'H8_target_crosses_gap_audit_only':bool(d['gap_cross_valid'][j]),
     'actual_abs_cumulative_H8_logreturn_bp':abs(8*actual*10000),'gross_directional_outcome_bp':gross,
     'train_majority_H8_sign':d['majority'],'train_mean_variance_baseline':d['train_mean_var']})

 df=pd.DataFrame(outrows)
 tables=build_metric_tables(df,list(FOLDS))
 positive=tables['positive'];nonpos=tables['nonpositive'];fulln=tables['fulln'];posn=tables['positive_n']
 grid=tables['grid'];nonposrows=tables['nonpositive_rows'];ratiorows=tables['ratio_rows'];overallrows=tables['overall_rows']
 majority={f:fold[f]['majority'] for f in FOLDS};trainmv={f:fold[f]['train_mean_var'] for f in FOLDS}

 # Separate exact fold/full H8 baseline summary.
 base_rows=[]
 for f in list(FOLDS)+['ALL']:
  b=df if f=='ALL' else df[df.outer_fold==f]
  h,rate,n,ba=hit_stats(b.actual_sign_H8.to_numpy(dtype=np.int8),b.pred_sign_H8.to_numpy(dtype=np.int8))
  base_rows.append({'fold':f,'origins':len(b),'h8_actual_zero_n':int((b.actual_sign_H8==0).sum()),'h8_prediction_zero_n':int((b.pred_sign_H8==0).sum()),'hits':h,'eval_n':n,'hit_rate':rate,'balanced_accuracy':ba,'raw_variance_positive_n':int((b.y_var_prediction_H8_raw>0).sum()),'raw_variance_negative_n':int((b.y_var_prediction_H8_raw<0).sum()),'raw_variance_exact_zero_n':int((b.y_var_prediction_H8_raw==0).sum())})

 # Save the primary mean/variance results before the supplemental Gaussian band diagnostic.
 df.sort_values(['outer_fold','origin_row']).to_csv(out/'outer_predictions_mean_variance.csv',index=False,float_format='%.17g')
 write_csv(out/'mean_sigma_grid_metrics.csv',grid);write_csv(out/'raw_nonpositive_variance_metrics.csv',nonposrows);write_csv(out/'mean_sigma_ratio_metrics.csv',ratiorows);write_csv(out/'variance_overall_metrics.csv',overallrows)
 write_csv(out/'train_prediction_thresholds.csv',thresholds);write_csv(out/'variance_fit_index.csv',fit_records);write_csv(out/'coverage_and_baseline.csv',base_rows)
 np.savez_compressed(out/'variance_coefficients_and_scalers.npz',**params)

 # Minimal but complete primary report; full per-fold/cell CSVs are the detailed tables.
 report=['# EURUSD H1 H8 mean/variance joint utility — primary results','',f'Run `{runid}`. Exactly four variance-only Ridge fits; saved H8 mean forecasts reused unchanged. No post-July-3 rows read.','',
  'Variance target is ddof=0 dispersion of the next eight observed log returns around their realized mean. It is within-window dispersion, not uncertainty of the H8 mean. `sigma=sqrt(predicted variance)` applies only when raw variance>0. Raw<=0 remains separate; variance MAE/calibration uses max(0,raw).','',
  '### Full H8 mean baseline by fold','', '| Fold | n | H8 hits / n | hit rate | balanced accuracy | raw var positive / negative / exact zero |','|---|---:|---:|---:|---:|---:|']
 for r in base_rows:
  report.append(f"| {r['fold']} | {r['origins']} | {r['hits']}/{r['eval_n']} | {fmt(None if r['hit_rate'] is None else 100*r['hit_rate'],'.2f')}% | {fmt(None if r['balanced_accuracy'] is None else 100*r['balanced_accuracy'],'.2f')}% | {r['raw_variance_positive_n']}/{r['raw_variance_negative_n']}/{r['raw_variance_exact_zero_n']} |")
 report+=['','### Train-only cutoffs','', '| Fold | n train | abs mean q33/q67 cumulative bp | sigma q33/q67 | ratio q33/q67 | mean/sigma/ratio train bin counts |','|---|---:|---:|---:|---:|---|']
 for r in thresholds:
  report.append(f"| {r['fold']} | {r['n_train']} | {r['mean_q33_abs_cumulative_bp']:.4f}/{r['mean_q67_abs_cumulative_bp']:.4f} | {r['sigma_q33']:.8g}/{r['sigma_q67']:.8g} | {r['ratio_q33']:.4f}/{r['ratio_q67']:.4f} | {'/'.join(map(str,r['train_mean_bin_n']))}; {'/'.join(map(str,r['train_sigma_bin_n']))}; {'/'.join(map(str,r['train_ratio_bin_n']))} |")
 report+=['','Mean edges use all train predictions; sigma and ratio edges use positive raw variance train predictions only. Ties go to lower bin; repeated edges and empty cells remain unchanged. ALL bins are fold-relative, not common absolute thresholds.','',
  '### Aggregate positive-variance 3×3 grid','',
  'Coverage uses all validation origins; cells include raw variance>0 only. Gross directional outcome is actual cumulative H8 log return × sign(saved mean forecast), in bp; median/p10 is descriptive gross outcome, not profit or max adverse excursion.','',
  '| mean × sigma | n / full coverage | zero actual | buy/sell | hit rate (n) | balanced acc. | majority acc. (n) | pred abs mean median | sigma median | realized abs bp median/p90 | gross bp median/p10 | var MAE / train mean MAE | var skill | calib ratio |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in [x for x in grid if x['fold']=='ALL']:
  hit='—' if r['h8_hit_rate'] is None else f"{100*r['h8_hit_rate']:.1f}% ({r['h8_hits']}/{r['h8_direction_eval_n']})"
  ba='—' if r['h8_balanced_accuracy'] is None else f"{100*r['h8_balanced_accuracy']:.1f}%"
  maj='—' if r['train_majority_accuracy'] is None else f"{100*r['train_majority_accuracy']:.1f}% ({r['train_majority_eval_n']})"
  gross='—' if r['n']==0 else f"{r['gross_directional_outcome_bp_median']:.3f}/{r['gross_directional_outcome_bp_p10']:.3f}"
  absbp='—' if r['n']==0 else f"{r['realized_abs_H8_cumulative_bp_median']:.3f}/{r['realized_abs_H8_cumulative_bp_p90']:.3f}"
  varmae='—' if r['variance_mae'] is None else f"{r['variance_mae']:.8g}/{r['train_mean_variance_baseline_mae']:.8g}"
  skill='—' if r['variance_mae_skill'] is None else f"{r['variance_mae_skill']:.4f}"
  cal='—' if r['variance_mean_calibration_ratio'] is None else f"{r['variance_mean_calibration_ratio']:.4f}"
  report.append(f"| {r['cell']} | {r['n']}/{100*r['coverage_full_validation']:.1f}% | {r['h8_zero_actual_mean_count']} | {r['h8_buy_count']}/{r['h8_sell_count']} | {hit} | {ba} | {maj} | {fmt(r['predicted_abs_mean_per_return_median'])} | {fmt(r['predicted_sigma_median'])} | {absbp} | {gross} | {varmae} | {skill} | {cal} |")
 report += ['', '### Per-fold stability and sparse counts','', 'The complete fold/cell means, medians, H8 hit, gross outcome median/p10, variance MAE/calibration, zero counts and train-majority comparisons are in `mean_sigma_grid_metrics.csv`. Empty cells are preserved.', '', '| Fold | mean × sigma | n | H8 hit rate | gross median/p10 bp | variance MAE skill |','|---|---|---:|---:|---:|---:|']
 for r in grid:
  if r['fold']=='ALL':continue
  hit='—' if r['h8_hit_rate'] is None else f"{100*r['h8_hit_rate']:.1f}% ({r['h8_hits']}/{r['h8_direction_eval_n']})"
  gross='—' if r['n']==0 else f"{r['gross_directional_outcome_bp_median']:.3f}/{r['gross_directional_outcome_bp_p10']:.3f}"
  skill='—' if r['variance_mae_skill'] is None else f"{r['variance_mae_skill']:.4f}"
  report.append(f"| {r['fold']} | {r['cell']} | {r['n']} | {hit} | {gross} | {skill} |")
 report+=['','### Raw nonpositive variance subset','', 'Negative raw variance is invalid; exact zero is mathematically possible but does not demonstrate certainty. Full fold/mean-tercile details are in `raw_nonpositive_variance_metrics.csv`.','', '| Fold | raw<0 n | raw=0 n | raw<=0 n | full-fold coverage | H8 hit rate (n) |','|---|---:|---:|---:|---:|---:|']
 for f in list(FOLDS)+['ALL']:
  r=next(x for x in nonposrows if x['fold']==f and x['cell']=='all_mean_bins / raw_nonpositive')
  hit='—' if r['h8_hit_rate'] is None else f"{100*r['h8_hit_rate']:.1f}% ({r['h8_hits']}/{r['h8_direction_eval_n']})"
  report.append(f"| {f} | {r['raw_variance_negative_prediction_count']} | {r['raw_variance_exact_zero_prediction_count']} | {r['n']} | {100*r['coverage_full_validation']:.1f}% | {hit} |")
 report+=['','### Positive-variance abs-mean/sigma ratio terciles','', '| Fold | ratio group | n | H8 hit rate | gross median/p10 bp | variance MAE skill |','|---|---|---:|---:|---:|---:|']
 for r in ratiorows:
  hit='—' if r['h8_hit_rate'] is None else f"{100*r['h8_hit_rate']:.1f}% ({r['h8_hits']}/{r['h8_direction_eval_n']})"
  gross='—' if r['n']==0 else f"{r['gross_directional_outcome_bp_median']:.3f}/{r['gross_directional_outcome_bp_p10']:.3f}"
  skill='—' if r['variance_mae_skill'] is None else f"{r['variance_mae_skill']:.4f}"
  report.append(f"| {r['fold']} | {r['cell']} | {r['n']} | {hit} | {gross} | {skill} |")
 report+=['','No post-hoc cell is recommended. No IID significance or profitability is claimed. A normal distribution would be an extra assumption and requires checking coverage for the exact target; within-window variance is not H8-mean uncertainty.','',
  '- `mean_sigma_grid_metrics.csv`: full per-fold and aggregate 3×3 results.',
  '- `raw_nonpositive_variance_metrics.csv`: raw negative, exact-zero and combined nonpositive results.',
  '- `mean_sigma_ratio_metrics.csv`: positive-variance ratio terciles.',
  '- `variance_overall_metrics.csv`: full, positive, nonpositive, negative, and exact-zero variance cohorts.',
  '- `train_prediction_thresholds.csv`: fold-specific train-only cutoffs.',
  '- `outer_predictions_mean_variance.csv`: predictions, actual targets and audit fields separately.',
  '- `variance_fit_index.csv`, `variance_coefficients_and_scalers.npz`: four fitted variance heads.', '']
 (out/'primary_results_report.md').write_text('\n'.join(report),encoding='utf-8')

 # Save metadata and final summary before the separate Gaussian diagnostic.
 base_rate=next(r['hit_rate'] for r in base_rows if r['fold']=='ALL')
 summary={'run_id':runid,'status':'primary_complete_gaussian_diagnostic_pending','protocol_path':str(PROTOCOL.relative_to(REPO)),'protocol_sha256':sha(PROTOCOL),'source_prefix_sha256':source_run['prefix_sha256_verified'],'test_rows_read':0,'variance_fits_completed':4,'mean_models_refit':0,'outer_validation_origins':len(df),'positive_raw_variance_origins':len(positive),'raw_nonpositive_variance_origins':len(nonpos),'raw_negative_variance_origins':int((df.y_var_prediction_H8_raw<0).sum()),'raw_exact_zero_variance_origins':int((df.y_var_prediction_H8_raw==0).sum()),'h8_mean_full_validation_hit_rate':base_rate,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.perf_counter()-wall0,'thread_count':1,'device':'CPU','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}}
 (out/'run_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 (HERE/'latest_run.json').write_text(json.dumps({'run_id':runid,'results_dir':str(out.relative_to(REPO))},indent=2)+'\n')
 print(json.dumps(summary,indent=2,ensure_ascii=False));print('OUTPUT_DIR',out)

if __name__=='__main__':
 main()
