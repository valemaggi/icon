from pyswip import Prolog
import pandas as pd
import re
from sklearn.metrics import accuracy_score, recall_score
from models_results import write_results

df = pd.read_csv("./ragionamento/bin_dataset.csv", dtype=str)

# Funzione di normalizzazione
def normalize(val):
    return str(val).strip().lower().replace(" ", "_").replace("'", "")

# Predicati dalle regole Prolog
def extract_predicates_from_file(file_path):
    predicates = set()
    with open(file_path, 'r') as f:
        for line in f:
            if ":-" in line and "depression('1')" in line:
                body = line.split(":-")[1]
                matches = re.findall(r"(\w+)\(", body)
                for m in matches:
                    if m != "depression":
                        predicates.add(m)
    return sorted(predicates)

def rename_columns_to_match_predicates(df, predicates):
    new_columns = {}
    for col in df.columns:
        norm_col = normalize(col)
        for pred in predicates:
            if pred in norm_col:
                new_columns[col] = pred
                break
    df = df.rename(columns=new_columns)
    return df

# Estraggo fatti usando i predicati
def facts_from_row(row, predicates):
    facts = []
    for pred in predicates:
        if pred in row:
            val = normalize(row[pred])
            facts.append(f"{pred}('{val}')")
    return facts

prolog_file = "depression_rules.pl"
prolog = Prolog()
prolog.consult(prolog_file)

predicates = extract_predicates_from_file(prolog_file)
df = rename_columns_to_match_predicates(df, predicates)

def classify(row):
    for pred in predicates:
        prolog.retractall(f"{pred}(_)")
    for fact in facts_from_row(row, predicates):
        prolog.assertz(fact)
    result = list(prolog.query("depression(X)"))
    return int(result[0]["X"]) if result else -1

df["predicted_depression"] = df.apply(classify, axis=1)

# Valutazione
df_valid = df[df["predicted_depression"] != -1]
y_true = df_valid["Depression"].astype(int)
y_pred = df_valid["predicted_depression"]

accuracy = round(accuracy_score(y_true, y_pred), 3)
recall = round(recall_score(y_true, y_pred), 3)

print(df['predicted_depression'].value_counts())
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

write_results("./apprendimento_supervisionato/accuracy_recall.txt", "Prolog", accuracy, recall)