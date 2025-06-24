import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
import warnings


warnings.filterwarnings('ignore')

df = pd.read_csv('./datacleaning/student_depression_dataset.csv')

# Creo la cartella "images" se non esiste
os.makedirs('images', exist_ok=True)

# Grafici a barre
def create_percentage_bar_chart(column_name, percentages):
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(range(len(percentages)), percentages.values, 
                   color=["#8904f6", '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
    
    plt.title(f'Distribuzione Percentuale - Colonna: {column_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Valori', fontsize=12)
    plt.ylabel('Percentuale (%)', fontsize=12)
    
    plt.xticks(range(len(percentages)), percentages.index, rotation=45, ha='right')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(f'images/percentage_chart_{column_name}.png', dpi=300, bbox_inches='tight')

# Funzione per creare la cmap
def create_cmap():
    df2 = df.apply(lambda x: pd.factorize(x)[0])

    f, ax = plt.subplots(figsize=(10, 8))
    corr = df2.corr()
    sns.heatmap(corr,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0, vmax=1.0,
        square=True, ax=ax, fmt=".2f", annot=True)
    plt.savefig(f'images/cmap.png', dpi=f.dpi, bbox_inches='tight')
create_cmap()


print("=== ANALISI DATASET ===")
for col in df.columns:
    print(f"Unique values in: {df[col].value_counts()}\n")

print("Missing values")
print(df.isnull().sum()) 


def drop_columns(df):
    # Lista per tenere traccia delle colonne da eliminare
    columns_to_drop = ['id']

    columns_percentage_data = {}

    for col in df.columns:
        Percentuale = round(df[col].value_counts(normalize=True) * 100, 2)

        Percentuale_valore = Percentuale.iloc[0]
        print(f"Colonna '{col}'")
        print(f"Valore più alto: {Percentuale_valore}%")
        
        if Percentuale_valore > 99:
            columns_to_drop.append(col)
            print(f"Colonna '{col}' marcata per eliminazione")

            columns_percentage_data[col] = Percentuale

    # Elimino le colonne identificate
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Colonne eliminate: {columns_to_drop}")
        
        # Crea un grafico separato per ogni colonna eliminata
        for col_name, percentages in columns_percentage_data.items():
            create_percentage_bar_chart(col_name, percentages)
        
    else:
        print("Nessuna colonna da eliminare")
    
    return df
df = drop_columns(df)

# Rinomino le colonne
df = df.rename(columns={'Work/Study Hours':'Study Hours'})
df = df.rename(columns={'Have you ever had suicidal thoughts ?':'Suicidal thoughts'})

# Dizionari per mappare i valori testuali a numeri
sleep_mapping = {
    "'Less than 5 hours'": 1,
    "'5-6 hours'": 2,
    "'7-8 hours'": 3,
    "'More than 8 hours'": 4
}

df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)

dietary_mapping = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

df['Dietary Habits'] = df['Dietary Habits'].map(dietary_mapping)

df = df.replace( to_replace = {"No" : 0, "Yes" : 1})
df = df.replace( to_replace = {"Male" : 0, "Female" : 1})

# Tolgo valori non significativi
for column in df.columns:
    df = df[df[column]!='?']
    df = df[df[column]!='Others']

df = df.dropna()

df.to_csv('./datacleaning/dataset.csv', index=False)

# Trasformo da stringhe in interi
le = LabelEncoder()

df_models = df
df_models['City'] = le.fit_transform(df['City'])
df_models['Degree'] = le.fit_transform(df['Degree'])

df_models.to_csv('./apprendimento_supervisionato/dataset_model.csv', index=False)
