import pandas as pd
from itertools import combinations

df = pd.read_csv('./ragionamento/bin_dataset.csv')

# File in cui salvare i pattern
with open('./ragionamento/pattern_analysis.txt', 'w', encoding='utf-8') as f:
    
    min_occurrences = 1000

    # Funzione per trovare pattern di valori che si ripetono
    def find_repeated_patterns(df):

        patterns = {}

        print("\n=== PATTERN SINGOLE COLONNE ===")
        f.write("=== PATTERN SINGOLE COLONNE ===\n")
        
        for col in df.columns:
            value_counts = df[col].value_counts()
            frequent_values = value_counts[value_counts >= min_occurrences]
            if len(frequent_values) > 0:
                patterns[f"{col}_frequent"] = frequent_values
                print(f"\n{col}:")
                f.write(f"\n{col}:\n")
                # Ordina per occorrenze decrescenti
                for value, count in frequent_values.sort_values(ascending=False).items():
                    print(f"  '{value}': {count} occorrenze")
                    f.write(f"  '{value}': {count} occorrenze\n")

        print(f"\n=== PATTERN COMBINAZIONI (min {min_occurrences} occorrenze) ===")
        f.write(f"\n=== PATTERN COMBINAZIONI (min {min_occurrences} occorrenze) ===\n")
        
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                combination_counts = df.groupby([col1, col2]).size()
                frequent_combinations = combination_counts[combination_counts >= min_occurrences]
                if len(frequent_combinations) > 0:
                    patterns[f"{col1}_{col2}_combo"] = frequent_combinations
                    print(f"\n{col1} + {col2}:")
                    f.write(f"\n{col1} + {col2}:\n")

                    for (val1, val2), count in frequent_combinations.sort_values(ascending=False).items():
                        print(f"  '{val1}' + '{val2}': {count} occorrenze")
                        f.write(f"  '{val1}' + '{val2}': {count} occorrenze\n")
        
        
        print(f"\n=== PATTERN COMBINAZIONI 3 COLONNE (min {min_occurrences} occorrenze) ===")
        f.write(f"\n=== PATTERN COMBINAZIONI 3 COLONNE (min {min_occurrences} occorrenze) ===\n")
        
        for col_combo in combinations(df.columns, 3):
            combination_counts = df.groupby(list(col_combo)).size()
            frequent_combinations = combination_counts[combination_counts >= min_occurrences]
            if len(frequent_combinations) > 0:
                combo_name = "_".join(col_combo)
                patterns[f"{combo_name}_combo"] = frequent_combinations
                print(f"\n{' + '.join(col_combo)}:")
                f.write(f"\n{' + '.join(col_combo)}:\n")
                for pattern_values, count in frequent_combinations.sort_values(ascending=False).items():
                    values_str = "' + '".join([str(v) for v in pattern_values])
                    print(f"  '{values_str}': {count} occorrenze")
                    f.write(f"  '{values_str}': {count} occorrenze\n")
        
        print(f"\n=== RIGHE DUPLICATE COMPLETE ===")
        f.write(f"\n=== RIGHE DUPLICATE COMPLETE ===\n")
        duplicated_rows = df[df.duplicated()]
        if len(duplicated_rows) > 0:
            duplicate_counts = df.value_counts()
            frequent_duplicates = duplicate_counts[duplicate_counts >= 10]
            if len(frequent_duplicates) > 0:
                print("Righe che si ripetono:")
                f.write("Righe che si ripetono:\n")
                for i, (pattern, count) in enumerate(frequent_duplicates.sort_values(ascending=False).items()):
                    print(f"  Pattern {i+1}: {count} occorrenze")
                    print(f"    {pattern}")
                    f.write(f"  Pattern {i+1}: {count} occorrenze\n")
                    f.write(f"    {pattern}\n")
        else:
            print("Nessuna riga duplicata trovata")
            f.write("Nessuna riga duplicata trovata\n")
        
        return patterns

    patterns = find_repeated_patterns(df)

    print(f"\n=== PATTERN COMBINAZIONI 3 COLONNE (min {min_occurrences} occorrenze) ===")
    f.write(f"\n=== PATTERN COMBINAZIONI 3 COLONNE (min {min_occurrences} occorrenze) ===\n")

    for col_combo in combinations(df.columns, 3):
        combination_counts = df.groupby(list(col_combo)).size()
        frequent_combinations = combination_counts[combination_counts >= min_occurrences]
        if len(frequent_combinations) > 0:
            combo_name = "_".join(col_combo)
            patterns[f"{combo_name}_combo"] = frequent_combinations
            print(f"\n{' + '.join(col_combo)}:")
            print(f"  Trovate {len(frequent_combinations)} combinazioni frequenti")
            f.write(f"\n{' + '.join(col_combo)}:\n")
            f.write(f"  Trovate {len(frequent_combinations)} combinazioni frequenti\n")


    print("\n" + "="*60)
    print("=== RIASSUNTO: COLONNE PIÙ COINVOLTE NEI PATTERN ===")
    print("="*60)
    f.write("\n" + "="*60 + "\n")
    f.write("=== RIASSUNTO: COLONNE PIÙ COINVOLTE NEI PATTERN ===\n")
    f.write("="*60 + "\n")

    column_involvement = {}
    for col in df.columns:
        column_involvement[col] = 0

    for pattern_name, pattern_data in patterns.items():
        if 'frequent' in pattern_name:
            col_name = pattern_name.replace('_frequent', '')
            column_involvement[col_name] += len(pattern_data)
        elif 'combo' in pattern_name:
            cols_in_pattern = pattern_name.replace('_combo', '').split('_')
            for col in cols_in_pattern:
                if col in column_involvement:
                    column_involvement[col] += len(pattern_data)

    # Ordino le colonne per coinvolgimento
    sorted_columns = sorted(column_involvement.items(), key=lambda x: x[1], reverse=True)

    print("\nColonne ordinate per frequenza di coinvolgimento nei pattern:")
    f.write("\nColonne ordinate per frequenza di coinvolgimento nei pattern:\n")
    for col, involvement in sorted_columns:
        if involvement > 0:
            print(f"  {col}: coinvolta in {involvement} pattern frequenti")
            f.write(f"  {col}: coinvolta in {involvement} pattern frequenti\n")

    print(f"\n=== RELAZIONI PIÙ FORTI TRA COLONNE ===")
    f.write(f"\n=== RELAZIONI PIÙ FORTI TRA COLONNE ===\n")
    strongest_relations = []

    for pattern_name, pattern_data in patterns.items():
        if 'combo' in pattern_name and len(pattern_data) > 0:
            cols_involved = pattern_name.replace('_combo', '').split('_')
            num_patterns = len(pattern_data)
            strongest_relations.append((cols_involved, num_patterns))

    # Ordino per numero di pattern
    strongest_relations.sort(key=lambda x: x[1], reverse=True)

    print("\nRelazioni tra colonne ordinate per forza della relazione:")
    f.write("\nRelazioni tra colonne ordinate per forza della relazione:\n")
    for cols, num_patterns in strongest_relations[:10]:
        print(f"  {' + '.join(cols)}: {num_patterns} pattern comuni")
        f.write(f"  {' + '.join(cols)}: {num_patterns} pattern comuni\n")
