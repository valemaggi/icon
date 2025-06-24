import pandas as pd

df = pd.read_csv('./datacleaning/dataset.csv')

selected_columns = ['Age','Academic Pressure','CGPA','Study Satisfaction','Study Hours']

df['Age'] = df['Age'].apply(lambda x: 'liceale' if x < 20 else 'universitario' if x < 25 else 'ricercatore')
df['Academic Pressure'] = df['Academic Pressure'].apply(lambda x: 'bassa' if x < 3 else 'media' if x < 4 else 'alta')
df['CGPA'] = df['CGPA'].apply(lambda x: 'bassa' if x < 7 else 'media' if x < 9 else 'alta')
df['Study Satisfaction'] = df['Study Satisfaction'].apply(lambda x: 'bassa' if x < 3 else 'media' if x < 4 else 'alta')

df.to_csv('./ragionamento/bin_dataset.csv', index=False)