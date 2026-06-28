import pandas as pd
import os
p = os.path.join(os.path.dirname(__file__), '..', 'Epic 1 - Data Collection', 'creditcard.csv')
print('path:', p)
try:
    df = pd.read_csv(p, nrows=0)
    print('columns:', list(df.columns))
except Exception as e:
    print('ERROR', e)
