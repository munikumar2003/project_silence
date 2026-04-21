import sys; 
sys.path.insert(0, '.')
from data_ingestion import DataIngestor
from dataset_bias import DatasetBiasDetector
from model_bias import ModelBiasDetector
from mitigation import BiasMitigator
from report_generator import BiasAuditReport

# Pipeline
ingestor = DataIngestor()
ingestor.load_csv('loan_data.csv')
ingestor.detect_sensitive_columns()
ingestor.set_target_column('loan_approved')
profile= ingestor.profile_dataset()
df = ingestor.get_clean_df()
dataset_detector = DatasetBiasDetector(df=df, sensitive_cols=['gender', 'race'],target_col='loan_approved')
dataset_bias= dataset_detector.run_all_checks()

model_detector = ModelBiasDetector (df=df, sensitive_cols=['gender', 'race'],target_col='loan_approved')
model_detector.train_model('random_forest')
model_bias= model_detector.compute_fairness_metrics()
mitigator = BiasMitigator(df=df, sensitive_cols=['gender', 'race'], target_col='loan_approved')
r1 = mitigator.mitigate_reweighting()
r2 = mitigator.mitigate_resampling()
r3 = mitigator.mitigate_threshold_adjustment()
best = mitigator.get_best_strategy()

#Helpers (same as API)
def serialize(obj):
    if isinstance(obj, dict): return {k: serialize(v) for k,v in obj.items()}
    if isinstance(obj, list): return [serialize(i) for i in obj]
    if isinstance(obj, bool): return bool(obj)
    if hasattr(obj, 'item'): return obj.item()
    return obj

def risk_score(db, mb):
    scores = []
    for col, res in mb.items():
        di=res.get('disparate_impact_ratio', 1.0)
        dp=res.get('demographic_parity_difference', 0.0)
        scores.append(min((1-di)*50+ dp*100, 100))
    overall = round(sum(scores)/len(scores), 1) if scores else 0
    level='HIGH' if overall > 60 else 'MEDIUM' if overall > 30 else 'LOW'
    return {'score': overall, 'level': level}

def summary (db, mb, rs):
    sev = [c for c,r in mb.items() if r.get('severity')=='SEVERE']
    mod = [c for c,r in mb.items() if r.get('severity')=='MODERATE']
    total = sum(len(r.get('flags', ())) for r in mb.values()) + sum(len(r.get('flags', ())) for r in db.values())
    rec = ('URGENT: Do not deploy without mitigation.' 
           if rs['level']=='HIGH' 
           else 'WARNING: Apply mitigation before deployment.' 
           if rs['level']=='MEDIUM'
           else 'System appears relatively fair. Continue monitoring.')
    return {'risk_level': rs ['level'], 'risk_score': rs['score'], 'total_flags': total,
'severe_attributes': sev, 'moderate_attributes': mod, 'recommendation': rec}

rs =risk_score (dataset_bias, model_bias)
audit_results = {
'dataset_profile': profile,
'dataset_bias': serialize(dataset_bias),
'model_bias': serialize(model_bias),
'risk_score':rs,
'summary': summary (dataset_bias, model_bias, rs)
}

mitigation_results = {
    'mitigation_results':{
    'reweighting': serialize (r1),
    'resampling': serialize (r2),
    'threshold_adjustment': serialize(r3),
    },
    'recommended_strategy': best
}

#Generate PDF
report = BiasAuditReport(
audit_results=audit_results,
mitigation_results=mitigation_results,
org_name='Acme Financial Services',
output_path='bias_audit_report.pdf'
)
report.generate()
print('Done.')