import pandas as pd
df = pd.read_csv('core_shell_pairs_type1_in_range.csv')
print("Unique cores:", df['formula_core'].nunique())
print("Unique shells:", df['formula_shell'].nunique())
print(df['formula_shell'].value_counts().head(10))