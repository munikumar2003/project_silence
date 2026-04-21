import sys; sys.path.insert(0, '.')
from data_ingestion import DataIngestor
from mitigation import BiasMitigator

ingestor=DataIngestor()
ingestor.load_csv('loan_data.csv')
ingestor.detect_sensitive_columns()
ingestor.set_target_column('loan_approved')
ingestor.profile_dataset()

mitigator = BiasMitigator(
df=ingestor.get_clean_df(),
sensitive_cols=['gender', 'race'],
target_col='loan_approved'
)
mitigator.mitigate_reweighting()
mitigator.mitigate_resampling()
mitigator.mitigate_threshold_adjustment()
print('\nBest strategy:', mitigator.get_best_strategy())