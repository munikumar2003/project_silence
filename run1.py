import sys; sys.path.insert(0, '.')
from data_ingestion import DataIngestor
import json

ingestor = DataIngestor()
ingestor.load_csv('loan_data.csv')
ingestor.detect_sensitive_columns()
ingestor.set_target_column('loan_approved')
profile = ingestor.profile_dataset()

print('\n--- PROFILE SUMMARY ---')
print('Shape:', profile['shape'])
print('Missing values:', profile['missing_values'])
print('\nTarget distribution:', profile['target_distribution']['percentages'])
print('\nGroup outcome rates:')
for attr, rates in profile['group_outcome_rates'].items():
  print(f' {attr}: {rates}')