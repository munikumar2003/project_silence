import sys; sys.path.insert(0, ".")  # Ensure current dir is in path for imports
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(7)
n=800
df=pd.DataFrame({
'gender': np.random.choice(['Male','Female','Non-binary'],n,p=[0.52,0.43,0.05]),
'age': np.random.randint(22,60,n),
'ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'],n,p=[0.48,0.22,0.18,0.12]),
'years_exp': np.random.randint(0,20,n),
'skill_score': np.random.normal(65,15,n).clip (20,100).astype(int),})

base = (df['skill_score']-20)/80*0.6 + df['years_exp']/20*0.3
bias= np.where(df['gender']=='Female', -0.10,0) + np.where(df['ethnicity']=='Black',-0.12,0)
df['hired'] = (np.random.uniform (0,1,n) < (base+bias).clip(0,1)).astype(int)
df.to_csv('hiring.csv', index=False)

# User trains THEIR model with THEIR own encoding (which differs from ours)
X = df.drop('hired', axis=1).copy()
for col in ['gender', 'ethnicity']:
    le=LabelEncoder(); X[col]=le.fit_transform(X[col])
clf=RandomForestClassifier(n_estimators=50, random_state=7).fit(X, df['hired'])
joblib.dump(clf, 'hiring_model.pkl')

print('External model saved. Their encoding produces accuracy:',(clf.predict(X)==df['hired'].values).mean().round(3))
