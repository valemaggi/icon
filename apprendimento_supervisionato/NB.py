import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import GridSearchCV 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import json
from models_results import write_results

warnings.filterwarnings('ignore')

df = pd.read_csv('./apprendimento_supervisionato/dataset_model.csv')

# Carico prior dal file JSON
with open('./apprendimento_supervisionato/prior_probabilities.json') as f:
    priors = json.load(f)
    
prior_depression = [priors['depression'][label] for label in sorted(priors['depression'].keys())]

# Separazione features e target
X = df.drop(['Depression'], axis=1)
y = df['Depression']

# Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print('Model recall score with default hyperparameters: {0:0.4f}'. format(recall_score(y_test, y_pred)))

# GridSearch
param_grid = {
    'priors': [prior_depression]
}

grid = GridSearchCV(GaussianNB(), param_grid, cv=10, scoring='accuracy')
 
grid.fit(X_train, y_train)

print(grid.best_params_) 
 
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test) 

print('Model accuracy score after grid search: {0:0.4f}'. format(accuracy_score(y_test, grid_predictions)))
print('Model recall score after grid search: {0:0.4f}'. format(recall_score(y_test, grid_predictions)))

accuracy = round(accuracy_score(y_test, grid_predictions), 3)
recall = round(recall_score(y_test, grid_predictions), 3)

print(classification_report(y_test, grid_predictions))

# Matrice di confusione
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', square=True)
plt.title('Matrice di Confusione')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.savefig(f'./apprendimento_supervisionato/results/NB_matrix.png', bbox_inches='tight')

write_results("./apprendimento_supervisionato/accuracy_recall.txt", "NB", accuracy, recall)
