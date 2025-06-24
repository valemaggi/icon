import pandas as pd
from itertools import combinations
import json
import re

df = pd.read_csv('./ragionamento/bin_dataset.csv')

# Funzione di normalizzazione per i nomi delle colonne e valori
def normalize_column_name(name):
    return re.sub(r'\s+', '_', str(name).strip().lower().replace("?", ""))

def normalize_value(val):
    return re.sub(r'\s+', '_', str(val).strip().lower().replace("'", ""))

def find_patterns_with_confidence(
    df,
    max_size=3,
    min_support_count=2000,
    min_confidence=0.75,
    rule_target_col='Depression',
    rule_target_val='1'
):

    df.columns = [normalize_column_name(c) for c in df.columns]
    rule_target_col = normalize_column_name(rule_target_col)
    rule_target_val = normalize_value(rule_target_val)

    df_str = df.astype(str)
    results = []
    class_counts = {}

    for size in range(2, max_size + 1):
        for cols in combinations(df.columns, size):
            cols_list = list(cols)
            group = df_str[cols_list].groupby(cols_list).size()
            frequent = group[group >= min_support_count].sort_values(ascending=False)

            if not frequent.empty:
                top_patterns = frequent.head(10)
                for values, support_ab in top_patterns.items():
                    values = tuple(values) if isinstance(values, tuple) else (values,)

                    antecedent_cols = list(cols[:-1])
                    consequent_col = cols[-1]
                    antecedent_vals = values[:-1]
                    consequent_val = values[-1]

                    if len(antecedent_cols) == 1:
                        condition_a = df_str[antecedent_cols[0]] == antecedent_vals[0]
                    else:
                        condition_a = (df_str[antecedent_cols] == list(antecedent_vals)).all(axis=1)

                    support_a = condition_a.sum()
                    confidence = support_ab / support_a if support_a > 0 else 0

                    results.append({
                        "cols": list(cols),
                        "values": list(values),
                        "support": support_ab,
                        "confidence": round(confidence, 3),
                        "antecedent": list(zip(antecedent_cols, antecedent_vals)),
                        "consequent": (consequent_col, consequent_val)
                    })

                    if consequent_col == rule_target_col:
                        class_counts.setdefault(consequent_val, 0)
                        class_counts[consequent_val] += support_ab

    results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    prolog_rules = []
    unary_antecedents = set()

    for r in results:
        if (
            r['consequent'][0] == rule_target_col and
            r['consequent'][1] == rule_target_val and
            r['confidence'] >= min_confidence and
            len(r['antecedent']) == 1
        ):
            unary_antecedents.add(r['antecedent'][0])

    for r in results:
        if (
            r['consequent'][0] == rule_target_col and
            r['consequent'][1] == rule_target_val and
            r['confidence'] >= min_confidence
        ):
            antecedents = r['antecedent']

            if len(antecedents) > 1 and any(a in unary_antecedents for a in antecedents):
                continue

            ant_terms = ', '.join([
                f"{normalize_column_name(col)}('{normalize_value(val)}')" for col, val in antecedents
            ])
            cons_term = f"{normalize_column_name(r['consequent'][0])}('{normalize_value(r['consequent'][1])}')"
            rule = f"{cons_term} :- {ant_terms}."
            prolog_rules.append(rule)


    total = sum(class_counts.values())
    priors = {
        rule_target_col: {
            val: round(count / total, 5) for val, count in class_counts.items()
        }
    }

    pattern_groups = {}

    for r in results:
        if r['confidence'] >= min_confidence:
            col_key = ' + '.join(r['cols'])
            val_key = tuple(r['values'])
            pattern_groups.setdefault(col_key, []).append((val_key, r['support'], r['confidence']))

    # Stampo i pattern
    print("\nPattern significativi trovati:\n")
    for col_key, patterns in pattern_groups.items():
        print(f"{col_key}:")
        # Ordino per supporto decrescente
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for values, support, conf in sorted_patterns:
            value_str = ' + '.join([f"'{v}'" for v in values])
            print(f"  {value_str}: {support} occorrenze (confidence: {conf})")
        print()

    return prolog_rules, priors

rules, priors = find_patterns_with_confidence(df)

# Scrittura su file Prolog
with open("depression_rules.pl", "w") as f:
    f.write("% depression_rules.pl\n\n")
    for rule in rules:
        f.write(rule + '\n')
    f.write("\ndepression('0') :- \\+ depression('1').\n")

# Salvataggio dei prior in JSON
with open("./apprendimento_supervisionato/prior_probabilities.json", "w") as f:
    json.dump(priors, f, indent=4)

print("\nProbabilità a priori:")
for val, prob in priors['depression'].items():
    print(f"  {val}: {prob}")