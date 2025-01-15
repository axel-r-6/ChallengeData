import pandas as pd
from collections import Counter

# compte le nombre de fois que chaque eqt_code_cat apparait dans le dataset
df = pd.read_csv('y_train.csv')
counts = Counter(df['eqt_code_cat'])
sorted_counts = dict(sorted(counts.items()))

for eqt_code, count in sorted_counts.items():
    print(f'eqt_code_cat {eqt_code}: {count} fois')
