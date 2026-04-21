import sys; sys.path.insert(0, '.')
from data_ingestion import DataIngestor
from dataset_bias import DatasetBiasDetector
import json

ingestor=DataIngestor()
ingestor.load_csv('loan_data.csv')
ingestor.detect_sensitive_columns()
ingestor.set_target_column('loan_approved')
ingestor.profile_dataset()


# Run dataset bias detection
detector = DatasetBiasDetector(
    df=ingestor.get_clean_df(),
    sensitive_cols=['gender', 'race'],
    target_col='loan_approved'
)
results = detector.run_all_checks()
print('\n--- ALL FLAGS ---')
for attr, res in results.items():
  for flag in res['flags']:
    print(f'[{flag["severity"]}] {flag["type"]}: {flag["message"]}')