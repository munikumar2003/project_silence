import sys; sys.path.insert(0, ".")  # Ensure current dir is in path for imports
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

np.random.seed(7)
n = 800
df = pd.DataFrame({
    'gender': np.random.choice(['Male','Female','Non-binary'], n, p=[0.52,0.43,0.05]),
    'age': np.random.randint(22,60,n),
    'ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n, p=[0.48,0.22,0.18,0.12]),
    'years_exp': np.random.randint(0,20,n),
    'skill_score': np.random.normal(65,15,n).clip(20,100).astype(int),
})

base = (df['skill_score']-20)/80*0.6 + df['years_exp']/20*0.3
bias = np.where(df['gender']=='Female', -0.10, 0) + np.where(df['ethnicity']=='Black', -0.12, 0)
df['hired'] = (np.random.uniform(0,1,n) < (base+bias).clip(0,1)).astype(int)
df.to_csv('hiring.csv', index=False)

# Separate features and target
X = df.drop('hired', axis=1)
y = df['hired']

# Fix: Create a preprocessor that handles text columns gracefully
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['gender', 'ethnicity'])
    ],
    remainder='passthrough' # Keeps numerical columns (age, years_exp, skill_score) intact
)

# Fix: Bundle preprocessing and model into a singular Pipeline
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=7))
])

# Train the entire pipeline (including the encoding step)
clf_pipeline.fit(X, y)

# Save the unified pipeline object
joblib.dump(clf_pipeline, 'hiring_model.pkl')

print('Self-contained Pipeline model saved. Verification accuracy:', (clf_pipeline.predict(X) == y.values).mean().round(3))