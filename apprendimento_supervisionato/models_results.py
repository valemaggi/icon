import numpy as np 
import matplotlib.pyplot as plt 
import os

def write_results(file_path, label, accuracy, recall):
    new_line = f"{label}: {accuracy} {recall}\n"
    lines = []

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

    # Cerco se l'etichetta esiste già
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{label}:"):
            lines[i] = new_line
            found = True
            break

    # Se l'etichetta non esiste, la aggiungo
    if not found:
        lines.append(new_line)
    
    with open(file_path, "w") as f:
        f.writelines(lines)

def read_results(file_path, model):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith(f"{model}:"):
                try:
                    _, values = line.split(":")
                    accuracy_str, recall_str = values.strip().split()
                    return float(accuracy_str), float(recall_str)
                except ValueError:
                    raise ValueError(f"Formato non valido per la riga: {line.strip()}")
    raise ValueError(f"Modello '{model}' non trovato nel file.")

file_path = "./apprendimento_supervisionato/accuracy_recall.txt"
models = ['LR', 'NB', 'RF', 'SVC', 'Prolog']

accuracy = []
recall = []

for model in models:
    acc, rec = read_results(file_path, model)
    accuracy.append(acc)
    recall.append(rec)

# Grafico a barre
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

br1 = np.arange(len(accuracy)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

plt.bar(br1, accuracy, color ='blueviolet', width = barWidth, 
        edgecolor ='black', label ='accuracy') 
plt.bar(br2, recall, color ='hotpink', width = barWidth, 
        edgecolor ='black', label ='recall') 

plt.xlabel('Models', fontweight ='bold', fontsize = 15) 
plt.ylabel('Values', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(accuracy))], 
        ['LR', 'NB', 'RF', 'SVC', 'Prolog'])

plt.legend(loc = "upper left")
plt.savefig(f'./apprendimento_supervisionato/results/accuracy_recall_graph.png')