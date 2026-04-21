import sys; sys.path.insert(0,'.')
from run_audit import run_pipeline, serialize
result = run_pipeline(
    csv_path='loan_data.csv',
    target_col='loan_approved',
    sensitive_cols=['gender', 'race'],
    generate_pdf=False
)
print('Keys returned:', list(result.keys()))
print('Risk:', result['risk_score'])
print('Summary:', result['summary'])