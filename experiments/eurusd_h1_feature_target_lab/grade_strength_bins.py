#!/usr/bin/env python3
"""Grade saved EURUSD H8 outer predictions by fold-specific train-strength bins.

This script applies saved coefficients/scalers for in-sample train predictions;
there is no model fitting in this analysis.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, platform, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PROTOCOL=HERE/'analysis_protocol.json'
SOURCE=REPO/'experiments/eurusd_h1_horizon_agreement/results/eurusd_h1_horizon_agreement_v1_20261022T170250Z'
V3=REPO/'experiments/eurusd_h1_feature_target_selection_v3/results/eurusd_h1_feature_target_v3_20261022T151237Z'
GROUPS={'own_ohlc_log_ratios','same_bar_cross_ratios','observed_transition_gap'}
LABELS=['Very weak','Weak','Medium','Strong','Very strong']
FILTERS=['H4_H8_agreement','H8_H24_agreement','Unanimous_H4_H8_H24']


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv_gz(p:Path):
    with gzip.open(p,'rt',newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def direction_hit(actual,pred):
    actual=np.asarray(actual,dtype=np.int8); pred=np.asarray(pred,dtype=np.int8)
    use=(actual!=0)&(pred!=0)
    n=int(use.sum())
    if n==0: return 0,None,0,None
    a,p=actual[use],pred[use]
    hits=int((a==p).sum())
    acc=hits/n
    classes=set(a.tolist())
    ba=None
    if classes=={-1,1}:
        ba=float(np.mean([np.mean(p[a==-1]==-1),np.mean(p[a==1]==1)]))
    return hits,acc,n,ba


def quantile_bin(values,edges):
    # Right-closed intervals: exact edge ties remain in the lower bin.
    return np.searchsorted(np.asarray(edges),np.asarray(values),side='left').astype(np.int8)


def pct(value,denom):
    return float(value/denom) if denom else 0.0


def primary_stats(df,fold_label,bin_label,n_all,majority_by_fold):
    n=len(df)
    if n:
        actual=df.H8_actual_sign.to_numpy(dtype=np.int8)
        pred=df.H8_prediction_sign.to_numpy(dtype=np.int8)
        use=(actual!=0)&(pred!=0)
        hits,acc,eval_n,ba=direction_hit(actual,pred)
        maj=np.asarray([majority_by_fold[str(f)] for f in df.outer_fold],dtype=np.int8)
        mhits,macc,men,mba=direction_hit(actual,maj)
        actual_mag=df.H8_actual_abs_cumulative_logreturn_bp.to_numpy(dtype=float)
        pred_mag=df.H8_expected_abs_cumulative_logreturn_bp.to_numpy(dtype=float)
        q25,q50,q75,q90=np.quantile(actual_mag,[.25,.5,.75,.9],method='linear')
        pred_mean,pred_median=float(np.mean(pred_mag)),float(np.median(pred_mag))
        actual_mean,actual_median=float(np.mean(actual_mag)),float(np.median(actual_mag))
        zero_actual=int((actual==0).sum())
        zero_h8=int((pred==0).sum())
        up=int((pred==1).sum()); down=int((pred==-1).sum())
        actual_up=int((actual==1).sum()); actual_down=int((actual==-1).sum())
    else:
        hits=eval_n=mhits=men=zero_actual=zero_h8=up=down=actual_up=actual_down=0
        acc=ba=macc=mba=pred_mean=pred_median=actual_mean=actual_median=q25=q50=q75=q90=None
    return {
      'fold':fold_label,'bin':bin_label,'n':n,'validation_origins':n_all,'coverage':pct(n,n_all),
      'h8_zero_forecast_count':zero_h8,'h8_zero_actual_count':zero_actual,
      'h8_up_signal_count':up,'h8_down_signal_count':down,'actual_up_count':actual_up,'actual_down_count':actual_down,
      'direction_hit_count':hits,'direction_eval_n':eval_n,'direction_hit_rate':acc,'balanced_accuracy':ba,
      'train_majority_hit_count':mhits,'train_majority_eval_n':men,'train_majority_accuracy':macc,'train_majority_balanced_accuracy':mba,
      'predicted_abs_cumulative_logreturn_bp_mean':pred_mean,'predicted_abs_cumulative_logreturn_bp_median':pred_median,
      'realized_abs_cumulative_logreturn_bp_mean':actual_mean,'realized_abs_cumulative_logreturn_bp_median':actual_median,
      'realized_abs_cumulative_logreturn_bp_q25':float(q25) if n else None,
      'realized_abs_cumulative_logreturn_bp_q75':float(q75) if n else None,
      'realized_abs_cumulative_logreturn_bp_p90':float(q90) if n else None,
    }


def main():
    global SOURCE, V3, PROTOCOL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=SOURCE,
                        help='saved horizon-agreement run; must match the frozen input hashes')
    parser.add_argument('--v3-run',type=Path,default=V3,
                        help='saved v3 run with feature matrix, targets, and splits; must match the frozen input hashes')
    parser.add_argument('--protocol',type=Path,default=PROTOCOL,
                        help='frozen analysis protocol with hashes for the specified upstream artifacts')
    parser.add_argument('--output-root',type=Path,default=HERE/'results')
    args=parser.parse_args()
    SOURCE=args.source_run.resolve(); V3=args.v3_run.resolve(); PROTOCOL=args.protocol.resolve()
    output_root=args.output_root.resolve()
    started=time.perf_counter()
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    protocol_sha=sha256_file(PROTOCOL)
    # Verify every frozen source artifact before reading its content.
    source_hashes=protocol['source_artifact_hashes_sha256']
    for rel,expected in source_hashes.items():
        p=(V3/rel[3:]) if rel.startswith('v3/') else (SOURCE/rel)
        got=sha256_file(p)
        if got!=expected: raise RuntimeError(f'Input artifact hash mismatch: {p}')

    fit_rows=list(csv.DictReader((SOURCE/'fit_index.csv').open(newline='')))
    h8_fits={r['outer_fold']:r for r in fit_rows if int(r['H'])==8}
    if set(h8_fits)!={'F1','F2','F3','F4'}: raise RuntimeError('Missing saved H8 fold fit records.')
    catalog=json.loads((V3/'feature_catalog.json').read_text(encoding='utf-8'))
    feature_indices=np.asarray([i for i,s in enumerate(catalog['schema']['16']) if s['group'] in GROUPS],dtype=int)
    names=[catalog['names_by_W']['16'][i] for i in feature_indices]
    if len(feature_indices)!=21: raise RuntimeError(f'Expected fixed 21-feature core mask, got {len(feature_indices)}.')
    with np.load(V3/'feature_matrices.npz',allow_pickle=False) as z:
        X=np.asarray(z['X_W16'],dtype=np.float64)[:,feature_indices]
        valid=np.asarray(z['valid_W16'],dtype=bool)
    split_rows=read_csv_gz(V3/'split_membership.csv.gz')
    targets={}
    with gzip.open(V3/'target_manifest.csv.gz','rt',newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r['H'])==8 and r['target_valid_next_observed_H']=='True':
                targets[int(r['origin_row'])]={
                    'actual':float(r['y_mean_log_return_per_observed_transition']),
                    'target_ts':r['target_timestamp'],
                    'origin_ts':r['origin_bar_open_timestamp'],
                    'decision_ts':r['decision_timestamp'],
                    'gap_count':int(r['target_non_one_hour_transition_count']),
                    'crosses_gap':r['target_crosses_non_one_hour_transition']=='True'
                }
    saved=pd.read_csv(SOURCE/'outer_predictions.csv')
    saved['origin_row']=saved['origin_row'].astype(int)
    if len(saved)!=6163: raise RuntimeError(f'Expected 6163 saved outer predictions, got {len(saved)}.')
    params=np.load(SOURCE/'model_coefficients_and_scalers.npz',allow_pickle=False)
    edges_by_fold={}; train_counts={}; train_majority={}; replay_max_error={}
    threshold_rows=[]
    train_strength_rows=[]
    for fold in ['F1','F2','F3','F4']:
        rec=h8_fits[fold]; fit_id=int(rec['fit_id']); prefix=f'{fit_id:02d}_{fold}_H8'
        rows=[r for r in split_rows if r['split_id']=='outer' and r['outer_fold']==fold]
        train_idx=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='train'),dtype=np.int64)
        val_idx=np.asarray(sorted(int(r['origin_row']) for r in rows if r['partition']=='validation'),dtype=np.int64)
        th=hashlib.sha256(train_idx.astype('<i8').tobytes()).hexdigest()
        vh=hashlib.sha256(val_idx.astype('<i8').tobytes()).hexdigest()
        if th!=rec['train_origins_sha256'] or vh!=rec['validation_origins_sha256']:
            raise RuntimeError(f'Existing model fit origin hash mismatch for {fold}.')
        coef=np.asarray(params[f'{prefix}_coef'],dtype=float).ravel()
        intercept=float(np.asarray(params[f'{prefix}_intercept']).ravel()[0])
        xm=np.asarray(params[f'{prefix}_x_mean'],dtype=float).ravel()
        xs=np.asarray(params[f'{prefix}_x_scale'],dtype=float).ravel()
        ym=float(np.asarray(params[f'{prefix}_y_mean'],dtype=float).ravel()[0])
        ys=float(np.asarray(params[f'{prefix}_y_scale'],dtype=float).ravel()[0])
        if len(coef)!=len(feature_indices) or len(xm)!=len(feature_indices) or len(xs)!=len(feature_indices):
            raise RuntimeError(f'Saved model dimensions do not match W16/core in {fold}.')
        if np.any(xs==0) or ys==0: raise RuntimeError(f'Invalid saved scaler in {fold}.')
        pred_train=(((X[train_idx]-xm)/xs)@coef+intercept)*ys+ym
        pred_valid=(((X[val_idx]-xm)/xs)@coef+intercept)*ys+ym
        if not np.isfinite(pred_train).all() or not np.isfinite(pred_valid).all():
            raise RuntimeError(f'Non-finite reconstructed predictions in {fold}.')
        saved_fold=saved[saved.outer_fold==fold].set_index('origin_row').loc[val_idx]
        saved_h8=saved_fold['y_mean_prediction_H8'].to_numpy(dtype=float)
        max_err=float(np.max(np.abs(pred_valid-saved_h8)))
        replay_max_error[fold]=max_err
        if not np.allclose(pred_valid,saved_h8,rtol=1e-10,atol=1e-14):
            raise RuntimeError(f'Saved H8 coefficients/scalers do not reproduce validation predictions in {fold}; stopping.')
        train_strength=np.abs(pred_train)
        q=np.quantile(train_strength,[.2,.4,.6,.8],method='linear')
        edges_by_fold[fold]=q
        train_counts[fold]=len(train_idx)
        train_bins=quantile_bin(train_strength,q)
        trcounts=Counter(int(x) for x in train_bins)
        train_majority[fold]=int(rec['majority_H8_prediction_sign'])
        row={'fold':fold,'h8_fit_id':fit_id,'n_h24_purged_train_origins':len(train_idx),
             'train_in_sample_strength_n':len(train_strength),'train_majority_H8_sign':train_majority[fold],
             'train_majority_positive_count':int(rec['majority_train_H8_positive']),
             'train_majority_negative_count':int(rec['majority_train_H8_negative']),
             'validation_prediction_replay_max_abs_error':max_err}
        for j,lab in enumerate(['q20','q40','q60','q80']):
            row[f'{lab}_abs_mean_logreturn_per_observed_transition']=float(q[j])
            row[f'{lab}_abs_expected_cumulative_logreturn_bp']=float(q[j]*8*10000)
        for i,label in enumerate(LABELS): row[f'train_count_{label.lower().replace(" ","_")}']=int(trcounts.get(i,0))
        threshold_rows.append(row)
        for i,val in enumerate(train_strength):
            train_strength_rows.append({'fold':fold,'train_origin_row':int(train_idx[i]),'train_abs_H8_mean_logreturn':float(val),'train_abs_expected_H8_cumulative_logreturn_bp':float(val*8*10000),'strength_bin':LABELS[int(train_bins[i])]})

    run_id='eurusd_h1_prediction_strength_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=output_root/run_id
    out.mkdir(parents=True,exist_ok=False)
    # Bin saved outer validation predictions using only fold-specific train-derived edges.
    all_rows=[]
    for _,r in saved.iterrows():
        fold=str(r.outer_fold); q=edges_by_fold[fold]
        strength=abs(float(r.y_mean_prediction_H8))
        b=int(quantile_bin([strength],q)[0])
        actual=float(r.y_mean_actual_H8); pred=float(r.y_mean_prediction_H8)
        actual_cum=8*actual*10000.0; pred_cum=8*pred*10000.0
        target=targets[int(r.origin_row)]
        outrow={
            'outer_fold':fold,'origin_row':int(r.origin_row),
            'origin_bar_open_timestamp':r.origin_bar_open_timestamp,'decision_timestamp':r.decision_timestamp,
            'target_timestamp_H8':r.target_timestamp_H8,
            'strength_bin_index':b+1,'strength_bin':LABELS[b],
            'H8_prediction_mean_logreturn_per_observed_transition':pred,
            'H8_abs_prediction_mean_logreturn_per_observed_transition':strength,
            'H8_expected_signed_cumulative_logreturn_bp':pred_cum,
            'H8_expected_abs_cumulative_logreturn_bp':abs(pred_cum),
            'H8_actual_mean_logreturn_per_observed_transition':actual,
            'H8_actual_sign':int(r.actual_sign_H8),
            'H8_actual_signed_cumulative_logreturn_bp':actual_cum,
            'H8_actual_abs_cumulative_logreturn_bp':abs(actual_cum),
            'H8_actual_exact_zero':bool(int(r.actual_sign_H8)==0),
            'H4_prediction_sign':int(r.pred_sign_H4),'H8_prediction_sign':int(r.pred_sign_H8),'H24_prediction_sign':int(r.pred_sign_H24),
            'H4_prediction_exact_zero':bool(int(r.pred_sign_H4)==0),'H8_prediction_exact_zero':bool(int(r.pred_sign_H8)==0),'H24_prediction_exact_zero':bool(int(r.pred_sign_H24)==0),
            'train_majority_H8_sign':train_majority[fold],
            'H8_target_gap_crossing_audit_only':bool(target['crosses_gap']),
            'H8_target_non_one_hour_transition_count_audit_only':target['gap_count'],
            'agreement_H4_H8':bool(int(r.pred_sign_H4)!=0 and int(r.pred_sign_H8)!=0 and int(r.pred_sign_H4)==int(r.pred_sign_H8)),
            'agreement_H8_H24':bool(int(r.pred_sign_H8)!=0 and int(r.pred_sign_H24)!=0 and int(r.pred_sign_H8)==int(r.pred_sign_H24)),
            'unanimous_H4_H8_H24':bool(int(r.pred_sign_H4)!=0 and int(r.pred_sign_H8)!=0 and int(r.pred_sign_H24)!=0 and int(r.pred_sign_H4)==int(r.pred_sign_H8)==int(r.pred_sign_H24)),
        }
        all_rows.append(outrow)
    bins=pd.DataFrame(all_rows)
    bins['decision_timestamp']=pd.to_datetime(bins['decision_timestamp'],utc=True)
    bins['decision_weekday_utc']=bins['decision_timestamp'].dt.day_name()
    bins['decision_timestamp']=bins['decision_timestamp'].map(lambda x:x.isoformat().replace('+00:00','Z'))
    bins['target_timestamp_H8']=pd.to_datetime(bins['target_timestamp_H8'],utc=True).map(lambda x:x.isoformat().replace('+00:00','Z'))
    bins.to_csv(out/'outer_validation_strength_bins.csv',index=False,float_format='%.17g')
    pd.DataFrame(threshold_rows).to_csv(out/'train_strength_thresholds.csv',index=False,float_format='%.17g')
    pd.DataFrame(train_strength_rows).to_csv(out/'train_in_sample_strength_predictions.csv',index=False,float_format='%.17g')

    primary=[]
    for fold in ['F1','F2','F3','F4','ALL']:
        base=bins if fold=='ALL' else bins[bins.outer_fold==fold]
        n_all=len(base)
        for i,label in enumerate(LABELS):
            subset=base[base.strength_bin==label]
            majority_map=train_majority
            primary.append(primary_stats(subset,fold,label,n_all,majority_map))
    pd.DataFrame(primary).to_csv(out/'strength_bin_metrics.csv',index=False,float_format='%.17g')

    secondary=[]
    filter_col={'H4_H8_agreement':'agreement_H4_H8','H8_H24_agreement':'agreement_H8_H24','Unanimous_H4_H8_H24':'unanimous_H4_H8_H24'}
    for fold in ['F1','F2','F3','F4','ALL']:
        base=bins if fold=='ALL' else bins[bins.outer_fold==fold]
        for label in LABELS:
            cohort=base[base.strength_bin==label]
            for f_name,col in filter_col.items():
                accepted=cohort[cohort[col].astype(bool)]
                actual=accepted.H8_actual_sign.to_numpy(dtype=np.int8)
                pred=accepted.H8_prediction_sign.to_numpy(dtype=np.int8)
                hits,hit_rate,eval_n,ba=direction_hit(actual,pred)
                mag=accepted.H8_actual_abs_cumulative_logreturn_bp.to_numpy(dtype=float)
                if len(mag):
                    magmean=float(np.mean(mag)); magmedian=float(np.median(mag)); q75=float(np.quantile(mag,.75,method='linear')); p90=float(np.quantile(mag,.9,method='linear'))
                else: magmean=magmedian=q75=p90=None
                zeros_req={'H4_H8_agreement':['H4_prediction_exact_zero','H8_prediction_exact_zero'],
                           'H8_H24_agreement':['H8_prediction_exact_zero','H24_prediction_exact_zero'],
                           'Unanimous_H4_H8_H24':['H4_prediction_exact_zero','H8_prediction_exact_zero','H24_prediction_exact_zero']}[f_name]
                rejected=cohort[~cohort[col].astype(bool)]
                zero_rejected=int(rejected[zeros_req].any(axis=1).sum()) if len(rejected) else 0
                secondary.append({
                    'fold':fold,'strength_bin':label,'bin_origins':len(cohort),
                    'filter':f_name,'accepted_n':len(accepted),'coverage_within_bin':pct(len(accepted),len(cohort)),
                    'rejected_due_to_exact_zero_forecast':zero_rejected,
                    'accepted_h8_actual_zero_count':int((actual==0).sum()),
                    'accepted_h8_up_signal_count':int((pred==1).sum()),'accepted_h8_down_signal_count':int((pred==-1).sum()),
                    'h8_direction_hit_count':hits,'h8_direction_eval_n':eval_n,'h8_direction_hit_rate':hit_rate,'h8_balanced_accuracy':ba,
                    'accepted_predicted_abs_cumulative_logreturn_bp_mean':float(accepted.H8_expected_abs_cumulative_logreturn_bp.mean()) if len(accepted) else None,
                    'accepted_predicted_abs_cumulative_logreturn_bp_median':float(accepted.H8_expected_abs_cumulative_logreturn_bp.median()) if len(accepted) else None,
                    'accepted_realized_abs_cumulative_logreturn_bp_mean':magmean,'accepted_realized_abs_cumulative_logreturn_bp_median':magmedian,
                    'accepted_realized_abs_cumulative_logreturn_bp_q75':q75,'accepted_realized_abs_cumulative_logreturn_bp_p90':p90,
                })
    pd.DataFrame(secondary).to_csv(out/'agreement_filters_by_strength_bin.csv',index=False,float_format='%.17g')

    # Search scope was local saved artifacts only. No raw OHLC was in the source run.
    path_status={
      'status':'not_computed_raw_ohlc_unavailable_in_saved_local_artifacts',
      'requested_window':'next eight observed bars, excluding origin bar',
      'required_fields':['origin close','next-eight-bar high','next-eight-bar low'],
      'available_saved_artifacts':['derived W16/core feature matrix','target manifest with mean/variance/gap metadata','saved predictions','model coefficients/scalers','split membership'],
      'missing_fields':['raw OHLC close/high/low series'],
      'action':'No external retrieval or Library retry; primary strength tables completed without this optional diagnostic.',
      'interpretation_guard':'No both-side crossing, oscillation, or path-excursion claims are made.'
    }
    (out/'path_diagnostic_status.json').write_text(json.dumps(path_status,indent=2,ensure_ascii=False)+'\n')

    # No local raw OHLC file was found in the existing project artifact inventory.
    protocol_hash=sha256_file(PROTOCOL)
    run_summary={
      'run_id':run_id,'status':'complete_primary_strength_analysis_path_diagnostic_unavailable',
      'protocol_file':str(PROTOCOL.relative_to(REPO)),'protocol_sha256':protocol_hash,
      'source_horizon_agreement_run_id':'eurusd_h1_horizon_agreement_v1_20261022T170250Z',
      'source_prefix_sha256':'141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270',
      'outer_validation_origins':len(bins),'source_test_rows_read':0,'new_model_fits':0,
      'train_prediction_method':'saved H8 coefficients/intercept and scalers applied to existing H24-purged train inputs; no refit',
      'saved_validation_prediction_replay_max_abs_error_by_fold':replay_max_error,
      'training_quantile_edges_by_fold':{f:[float(x) for x in edges_by_fold[f]] for f in edges_by_fold},
      'source_artifact_hashes_sha256':source_hashes,
      'metrics_files':['train_strength_thresholds.csv','train_in_sample_strength_predictions.csv','outer_validation_strength_bins.csv','strength_bin_metrics.csv','agreement_filters_by_strength_bin.csv','path_diagnostic_status.json'],
      'limitations':['training predictions are in-sample, validation occupancy need not equal 20%','pooled bin labels are fold-relative train-percentile strata, not common absolute thresholds','historical explored folds, not independent confirmation','no IID significance/profitability claims','raw OHLC path diagnostics unavailable'],
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
      'elapsed_seconds':time.perf_counter()-started,
    }
    (out/'run_summary.json').write_text(json.dumps(run_summary,indent=2,ensure_ascii=False)+'\n')

    # Human-readable report; tables are fully reproduced in CSV for additional precision.
    threshold_df=pd.DataFrame(threshold_rows)
    ptab=pd.DataFrame(primary)
    stab=pd.DataFrame(secondary)
    report=[
      '# EURUSD H1 H8 prediction-strength grading',
      '',
      f"Run: `{run_id}`. No model was fit; all H8 training predictions were reconstructed from saved coefficients/scalers. No test rows after July 3 were read.",
      '',
      '## Frozen method',
      '',
      'Strength is `abs(predicted mean H8 log return per observed transition)`. Displayed cumulative magnitude is `abs(8 × mean × 10,000)` in bp of log return, not pips. Per-fold q20/q40/q60/q80 edges came from that fold’s in-sample predictions on H24-purged training origins and were applied unchanged to its outer validation predictions. Validation occupancy is not expected to be 20%. Exact threshold ties go to the lower bin; repeated edges remain repeated and may leave bins empty. Pooled bins are fold-relative percentile strata, not a common absolute bp scale.',
      '',
      'Zero H8 forecasts and exact-zero realized outcomes are counted separately. Zero forecasts abstain and are excluded from directional accuracy; zero realized targets are separately counted and excluded. Direction prediction is the same saved H8 prediction. Training majority is the existing per-fold majority of nonzero H8 train labels, scored on the same actual-nonzero subset.',
      '',
      '## Per-fold train thresholds',
      '',
      '| Fold | Train n | q20 bp | q40 bp | q60 bp | q80 bp | Train bin counts (very weak → very strong) |',
      '|---|---:|---:|---:|---:|---:|---|'
    ]
    for _,r in threshold_df.iterrows():
      counts='/'.join(str(int(r[f'train_count_{x}'])) for x in ['very_weak','weak','medium','strong','very_strong'])
      report.append(f"| {r['fold']} | {int(r['n_h24_purged_train_origins'])} | {r['q20_abs_expected_cumulative_logreturn_bp']:.4f} | {r['q40_abs_expected_cumulative_logreturn_bp']:.4f} | {r['q60_abs_expected_cumulative_logreturn_bp']:.4f} | {r['q80_abs_expected_cumulative_logreturn_bp']:.4f} | {counts} |")
    report += ['', 'Threshold edges in both raw log-return units and bp are in `train_strength_thresholds.csv`; the training predictions are in-sample.']
    report += ['', '## Primary validation bins', '', 'Coverage is bin count divided by all outer validation origins in that fold (or all folds for ALL). Predicted/realized magnitudes below are absolute cumulative H8 log returns in bp. `hit` and majority accuracy exclude zero realized H8 targets; zero forecasts are separately counted. Balanced accuracy is blank unless both actual direction classes occur.', '']
    report += ['| Fold | Bin | n / coverage | zero pred / zero actual | up / down signals | H8 hit / n | balanced acc. | train majority / n | pred magnitude mean / median bp | realized magnitude mean / median / q75 / p90 bp |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary:
      majority='—' if r['train_majority_accuracy'] is None else f"{r['train_majority_accuracy']:.3f} / {r['train_majority_eval_n']}"
      hit='—' if r['direction_hit_rate'] is None else f"{r['direction_hit_rate']:.3f} ({r['direction_hit_count']}/{r['direction_eval_n']})"
      ba='—' if r['balanced_accuracy'] is None else f"{r['balanced_accuracy']:.3f}"
      pm='—' if r['n']==0 else f"{r['predicted_abs_cumulative_logreturn_bp_mean']:.3f} / {r['predicted_abs_cumulative_logreturn_bp_median']:.3f}"
      rm='—' if r['n']==0 else f"{r['realized_abs_cumulative_logreturn_bp_mean']:.3f} / {r['realized_abs_cumulative_logreturn_bp_median']:.3f} / {r['realized_abs_cumulative_logreturn_bp_q75']:.3f} / {r['realized_abs_cumulative_logreturn_bp_p90']:.3f}"
      report.append(f"| {r['fold']} | {r['bin']} | {r['n']} / {100*r['coverage']:.1f}% | {r['h8_zero_forecast_count']} / {r['h8_zero_actual_count']} | {r['h8_up_signal_count']} / {r['h8_down_signal_count']} | {hit} | {ba} | {majority} | {pm} | {rm} |")
    report += ['', '## Agreement filters, same H8 strength bins', '', 'Coverage is accepted rows divided by all rows in that fold/bin. The H8 direction remains unchanged. These are descriptive selections, not new models or thresholds.', '']
    report += ['| Fold | Bin | H4/H8 cov.; hit; actual abs median bp | H8/H24 cov.; hit; actual abs median bp | Unanimous cov.; hit; actual abs median bp |','|---|---|---:|---:|---:|']
    secdf=pd.DataFrame(secondary)
    for fold in ['F1','F2','F3','F4','ALL']:
      for label in LABELS:
        parts=[]
        for name in FILTERS:
          rr=secdf[(secdf.fold==fold)&(secdf.strength_bin==label)&(secdf['filter']==name)].iloc[0]
          cov=f"{100*rr.coverage_within_bin:.1f}%"
          hit='—' if pd.isna(rr.h8_direction_hit_rate) else f"{100*rr.h8_direction_hit_rate:.1f}%"
          mag='—' if pd.isna(rr.accepted_realized_abs_cumulative_logreturn_bp_median) else f"{rr.accepted_realized_abs_cumulative_logreturn_bp_median:.3f}"
          parts.append(f"{cov}; {hit}; {mag}")
        report.append(f"| {fold} | {label} | {parts[0]} | {parts[1]} | {parts[2]} |")
    report += ['', 'The hypothesis that a weaker H8 mean forecast implies a smaller realized move is evaluated by realized absolute-magnitude distributions in the primary table, not by selecting a best bin. Small mean returns can reflect cancellation of larger opposite-signed moves.', '', '## Optional path diagnostics', '', 'Not computed: the saved permitted-prefix artifacts contain derived features, targets, predictions, and parameters, but no raw OHLC close/high/low series. The next-eight-observed-bar both-side crossing and excursion measures therefore cannot be reconstructed; no external retrieval was attempted. Even if available, visiting both sides of origin close would not establish swing order or oscillation.', '', '## Limits and artifacts', '', 'This is an exploratory historical analysis on previously examined folds; the pooled “strong” bin is relative to each fold’s training prediction distribution. No strategy recommendation, probability-confidence claim, statistical significance, or profitability claim is made. Bid/spread data are incomplete.', '', '- `train_strength_thresholds.csv`: exact train-derived edges and train-bin occupancy.', '- `train_in_sample_strength_predictions.csv`: saved-coefficient train predictions used only to set edges.', '- `outer_validation_strength_bins.csv`: origin-level bins, predictions, actuals, signs, gap audit, and filter flags.', '- `strength_bin_metrics.csv`: complete primary per-fold and pooled table.', '- `agreement_filters_by_strength_bin.csv`: complete secondary pair/unanimity table.', '- `path_diagnostic_status.json`: path-data availability blocker.', '']
    (out/'results_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(run_summary,indent=2,ensure_ascii=False))
    print('OUTPUT_DIR',out)

if __name__=='__main__': main()
